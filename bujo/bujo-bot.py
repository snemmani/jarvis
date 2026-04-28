import os
import sys

# Ensure the project root is on sys.path so `bujo.*` imports work when this
# script is run directly (VS Code / debugpy adds the script dir, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import io
import json
import logging
import queue as _queue_mod
import re
import ssl
import uuid
from datetime import datetime

from fpdf import FPDF, XPos, YPos

import telegram
import wolframalpha
from telegram import Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_react_agent
from apscheduler.triggers.cron import CronTrigger

from bujo.base import (
    OPENAI_MODEL,
    TELEGRAM_TOKEN,
    expenses_model,
    mag_model,
    llm,
    check_authorization,
    WOLFRAM_APP_ID,
    PC_MAC_ADDRESS,
    BROADCAST_IP,
    WAKE_RELAY_URL,
    scheduler,
    CHAT_ID,
    openai_model,
    TEXT_TO_SPEECH_MODEL,
    portfolio_transactions_model,
    NOIP_USERNAME,
    NOIP_PASSWORD,
    NOIP_HOSTNAME,
    RESEARCH_MODEL,
)
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from bujo.portoflio.manage import PortfolioManager
from bujo.analytics.charts import spending_pie_chart, spending_bar_chart
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",
    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",
    "1. Expenses Tool – If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool – If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Portfolio_Tool – If I talk about portfolio transactions (buying or selling stocks/shares, recording a trade, listing my portfolio transactions), use this tool.",
    "4. Wolfram_Alpha_Tool – If I ask about current events, latest news, mathematical questions, astronomical questions, conversions between units, flight ticket fares, nutrition information of food, stock prices, use this tool.",
    "5. Translation_Tool – If I ask for translation:\n   - And I do not specify source/target languages, assume English to Sanskrit.\n   - If I do specify the languages, translate accordingly.",
    "6. Expense_Analytics_Tool – If I ask for expense charts, spending breakdown, category analysis, daily trend, or any analytics/visualisation, use this tool. Pass a JSON string with 'start_date' (YYYY-MM-DD), 'end_date' (YYYY-MM-DD, exclusive upper bound), and 'chart_type' ('pie' for category breakdown, 'bar' for daily trend). Example: {\"start_date\": \"2025-04-01\", \"end_date\": \"2025-05-01\", \"chart_type\": \"pie\"}.",
    "MOST IMPORTANT INSTRUCTIONS:",
    "Always call the appropriate tool based on my latest message. You must never answer directly without invoking a tool first.",
    "When sending a response (either to the LLM for summarization or to me), always return it as a string, not as a JSON object.",
    "Always return tool results in markdown format for readability.",
    "Separate individual items in the tool results with new lines.",
    "Use emojis wherever appropriate to make the response more friendly and visually appealing. 🎯🧮📅💸📰🌐🔢📊🖼️🔤",
]

# Configure logging with error handling for closed streams
class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that ignores ValueError from closed streams."""
    def emit(self, record):
        try:
            super().emit(record)
        except ValueError as e:
            if "I/O operation on closed file" in str(e):
                pass  # Ignore closed file errors during shutdown
            else:
                raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[SafeStreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_TG_MAX = 4096  # Telegram hard limit per message


async def _send_long(reply_func, text: str, **kwargs) -> None:
    """Send text, splitting into multiple messages if it exceeds Telegram's 4096-char limit.

    Splits preferentially on newlines so markdown blocks stay intact.
    """
    while text:
        if len(text) <= _TG_MAX:
            await reply_func(text, **kwargs)
            return
        split_at = text.rfind("\n", 0, _TG_MAX)
        if split_at <= 0:
            split_at = _TG_MAX
        await reply_func(text[:split_at], **kwargs)
        text = text[split_at:].lstrip("\n")


expense_manager  = ExpenseManager(expenses_model, mag_model)
mag_manager      = MagManager(mag_model)
portfolio_manager = PortfolioManager(portfolio_transactions_model)

wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)

# Persistent memory for the outer agent; survives agent reconstruction each message.
_main_memory = MemorySaver()

# Per-user Claude CLI session IDs — shared across /claudeApi and /portfolioSuggest.
_claude_sessions: dict[int, str] = {}
_research_input_queues: dict[int, _queue_mod.Queue] = {}

GRAHAM_PROMPT_PATH = "/home/bot/ai-prompts/GrahamPrompt.md"


async def _keep_typing(chat_id: int, bot, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, telegram.constants.ChatAction.TYPING)
        except Exception:
            pass
        await asyncio.sleep(4)


async def _run_claude(
    prompt: str,
    user_id: int,
    allowed_tools: str | None = None,
    timeout: int = 120,
) -> str:
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if allowed_tools:
        cmd += ["--allowedTools", allowed_tools]
    session_id = _claude_sessions.get(user_id)
    if session_id:
        cmd += ["--resume", session_id]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/app",
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return "⏱️ Request timed out. Please try again."

    raw = stdout.decode().strip()
    try:
        data = json.loads(raw)
        if data.get("session_id"):
            _claude_sessions[user_id] = data["session_id"]
        if data.get("is_error"):
            _claude_sessions.pop(user_id, None)
            return f"❌ Claude error: {data.get('result', 'unknown error')}"
        return data.get("result", "") or stderr.decode().strip() or "No output."
    except json.JSONDecodeError:
        return raw or stderr.decode().strip() or "No output."


def _report_to_pdf(
    text: str,
    title: str = "Portfolio Research Report",
    citations: list[str] | None = None,
    usage: dict | None = None,
) -> bytes:
    def _sanitise(s: str) -> str:
        s = (
            s.replace("₹", "Rs.")
            .replace("━", "=")
            .replace("─", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "--")
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "-")
            .replace("\u2026", "...")
        )
        # Strip control characters except newline/tab
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
        return s.encode("latin-1", errors="replace").decode("latin-1")

    safe = _sanitise(text)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitise(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y %I:%M %p IST')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    for line in safe.split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        try:
            if stripped.startswith("### "):
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 6, stripped[4:])
            elif stripped.startswith("## "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 7, stripped[3:])
            elif stripped.startswith("# "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.multi_cell(0, 8, stripped[2:])
            else:
                pdf.set_font("Helvetica", "", 9)
                clean = re.sub(r'\*+([^*]+)\*+', r'\1', stripped)
                clean = re.sub(r'`([^`]+)`', r'\1', clean)
                pdf.multi_cell(0, 5, clean)
        except Exception:
            pass  # skip any unprintable line rather than crashing the whole PDF

    if citations:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, "Sources & References")
        pdf.set_font("Helvetica", "", 8)
        for i, url in enumerate(citations, 1):
            safe_url = url.encode("latin-1", errors="replace").decode("latin-1")
            # Wrap long URLs to prevent FPDFException
            try:
                pdf.multi_cell(0, 5, f"[{i}] {safe_url}")
            except Exception:
                # If URL is too long, truncate it
                max_len = 100
                truncated = safe_url[:max_len] + "..." if len(safe_url) > max_len else safe_url
                try:
                    pdf.multi_cell(0, 5, f"[{i}] {truncated}")
                except Exception:
                    # Last resort: skip this citation
                    pass

    if usage:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, "Research Statistics")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, f"Model: {usage.get('model', RESEARCH_MODEL)}")
        pdf.multi_cell(0, 5, f"Input tokens:  {usage.get('prompt_tokens', 0):,}")
        pdf.multi_cell(0, 5, f"Output tokens: {usage.get('completion_tokens', 0):,}")
        pdf.multi_cell(0, 5, f"Total tokens:  {usage.get('total_tokens', 0):,}")
        if usage.get("total_cost"):
            pdf.multi_cell(0, 5, f"Estimated cost: ${usage['total_cost']:.4f}")

    return bytes(pdf.output())


class _TelegramProgressCallback(BaseCallbackHandler):
    """Sends agent tool-call steps as Telegram messages so the user sees live progress."""

    def __init__(self, loop: asyncio.AbstractEventLoop, bot, chat_id: int):
        super().__init__()
        self._loop = loop
        self._bot = bot
        self._chat_id = chat_id
        self._step = 0

    def _send(self, text: str):
        asyncio.run_coroutine_threadsafe(
            self._bot.send_message(chat_id=self._chat_id, text=text, parse_mode="Markdown"),
            self._loop,
        )

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._step += 1
        tool = serialized.get("name", "tool")
        preview = str(input_str)[:200].replace("`", "'")
        self._send(f"🔍 *Step {self._step} — {tool}*\n`{preview}`")

    def on_tool_end(self, output, **kwargs):
        preview = re.sub(r"[*_`\[\]]", "", str(output)[:300])
        self._send(f"📋 *Result preview:* {preview}")

    def on_agent_finish(self, finish, **kwargs):
        self._send("✅ *Research complete — generating PDF...*")


_RESEARCH_SYSTEM_PROMPT = (
    "You are a deep financial research analyst. When given a portfolio and research instructions, "
    "you MUST perform multiple rounds of web searches to gather current data. "
    "For each holding: search for current price, recent earnings, analyst ratings, and news. "
    "Cross-reference multiple sources. Synthesize a comprehensive, citation-rich report. "
    "Never fabricate data — only use what you find via search. "
    "Structure the report with clear sections: Executive Summary, Per-Holding Analysis, "
    "Portfolio Risk Assessment, and Recommendations."
)


async def _run_portfolio_research(
    user_prompt: str,
    system_prompt: str = _RESEARCH_SYSTEM_PROMPT,
    timeout: int = 1800,
    bot=None,
    chat_id: int | None = None,
    user_id: int | None = None,
) -> tuple[str, list[str], dict]:
    """Run portfolio research using OpenAI's native web search and code interpreter tools."""
    
    logger.info("portfolio_research: starting with OpenAI native tools, model=%s", RESEARCH_MODEL)
    
    # Send progress updates to Telegram
    async def send_progress(message: str):
        if bot and chat_id:
            try:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            except Exception as e:
                logger.warning("Failed to send progress update: %s", e)
    
    await send_progress("🔍 Starting research with web search and analysis tools...")
    
    try:
        # Create assistant with native tools
        assistant = openai_model.beta.assistants.create(
            name="Portfolio Research Analyst",
            instructions=system_prompt,
            model=RESEARCH_MODEL,
            tools=[
                {"type": "web_search"},
                {"type": "code_interpreter"}
            ]
        )
        
        # Create thread
        thread = openai_model.beta.threads.create()
        
        # Add user message
        openai_model.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt
        )
        
        await send_progress("📊 Analyzing portfolio data...")
        
        # Run the assistant
        run = openai_model.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Poll for completion with progress updates
        last_status = None
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                openai_model.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                raise asyncio.TimeoutError("Research timed out")
            
            run_status = openai_model.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            # Send status updates
            if run_status.status != last_status:
                status_messages = {
                    "queued": "⏳ Research queued...",
                    "in_progress": "🔬 Research in progress...",
                    "requires_action": "❓ Needs additional input...",
                    "completed": "✅ Research completed!",
                    "failed": "❌ Research failed",
                    "cancelled": "🚫 Research cancelled",
                    "expired": "⏰ Research expired"
                }
                if run_status.status in status_messages:
                    await send_progress(status_messages[run_status.status])
                last_status = run_status.status
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                error_msg = getattr(run_status, "last_error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Research {run_status.status}: {error_msg}")
            elif run_status.status == "requires_action":
                # Handle function calls if needed (e.g., human_input)
                required_action = run_status.required_action
                if required_action and required_action.type == "submit_tool_outputs":
                    tool_outputs = []
                    for tool_call in required_action.submit_tool_outputs.tool_calls:
                        if tool_call.function.name == "human_input":
                            # Ask user for input
                            args = json.loads(tool_call.function.arguments)
                            question = args.get("question", "Need clarification")
                            
                            await send_progress(f"❓ *Agent needs input:*\n{question}\n\n_Reply to continue research._")
                            
                            q = _research_input_queues.get(user_id) if user_id else None
                            if q:
                                try:
                                    answer = q.get(timeout=300)
                                except _queue_mod.Empty:
                                    answer = "No response — proceed with available data"
                            else:
                                answer = "Unable to get user input — proceed with available data"
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": answer
                            })
                    
                    if tool_outputs:
                        openai_model.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
            
            await asyncio.sleep(2)
        
        # Get the messages
        messages = openai_model.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc"
        )
        
        # Extract the assistant's response
        assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
        if not assistant_messages:
            raise RuntimeError("No response from assistant")
        
        final_message = assistant_messages[-1]
        
        # Extract text content
        content_parts = []
        for content_block in final_message.content:
            if content_block.type == "text":
                content_parts.append(content_block.text.value)
        
        output = "\n\n".join(content_parts)
        
        # Extract citations/sources from annotations
        sources = []
        for content_block in final_message.content:
            if content_block.type == "text" and hasattr(content_block.text, "annotations"):
                for annotation in content_block.text.annotations:
                    if hasattr(annotation, "url"):
                        url = annotation.url
                        if url and url not in sources:
                            sources.append(url)
        
        # Get usage statistics
        usage = {
            "model": RESEARCH_MODEL,
            "prompt_tokens": run_status.usage.prompt_tokens if hasattr(run_status, "usage") else 0,
            "completion_tokens": run_status.usage.completion_tokens if hasattr(run_status, "usage") else 0,
            "total_tokens": run_status.usage.total_tokens if hasattr(run_status, "usage") else 0,
        }
        
        logger.info("portfolio_research: completed, sources=%d, tokens=%d", 
                   len(sources), usage["total_tokens"])
        
        # Cleanup
        try:
            openai_model.beta.assistants.delete(assistant.id)
            openai_model.beta.threads.delete(thread.id)
        except Exception as e:
            logger.warning("Failed to cleanup assistant/thread: %s", e)
        
        return output, sources, usage
        
    except Exception as e:
        logger.error("portfolio_research: failed with error: %s", e, exc_info=True)
        raise


# Static tools (no Telegram context needed)
_static_tools = [
    Tool(
        name="Expenses_Interaction",
        func=expense_manager.agent_expenses,
        description="Use this tool to add or list expenses.",
        return_direct=True,
    ),
    Tool(
        name="MAG_interaction",
        func=mag_manager.agent_mag,
        description="Use this tool to manage MAG (calendar).",
        return_direct=True,
    ),
    Tool(
        name="Portfolio_Tool",
        func=portfolio_manager.agent_portfolio,
        description="Use this tool to record or list portfolio transactions (buy/sell stocks).",
        return_direct=True,
    ),
    Tool(
        name="Translation_Tool",
        func=lambda x: "This tool must be awaited",
        description="Use this tool to translate from one language to another.",
        coroutine=lambda x: llm.ainvoke([HumanMessage(x)]),
    ),
]


async def _make_wolfram_tool(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tool:
    async def query_tool(query: str) -> str:
        logger.info("Querying Wolfram Alpha for: %s", query)
        try:
            response = await wolfram_client.aquery(query)
            if hasattr(response, "pod"):
                for pod in response["pod"]:
                    subpod = pod["subpod"]
                    if isinstance(subpod, list):
                        for item in subpod:
                            await context.bot.send_photo(
                                chat_id=update.effective_chat.id,
                                photo=item["img"]["@src"],
                                caption=item["@title"],
                            )
                    else:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=subpod["img"]["@src"],
                            caption=pod["@title"],
                        )
            return "Response completed."
        except Exception as e:
            logger.error("Error querying Wolfram Alpha: %s", e)
            return "Wolfram Alpha cannot handle this request, LLM should handle this if it can."

    return Tool(
        name="Wolfram_Alpha_Tool",
        description="Wolfram_Alpha_Tool for complex or realtime queries.",
        func=query_tool,
        coroutine=query_tool,
    )


async def _make_analytics_tool(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tool:
    async def analytics_tool(query: str) -> str:
        logger.info("Expense analytics request: %s", query)
        try:
            params = json.loads(query.replace("```json", "").replace("```", "").strip())
        except Exception:
            return "Invalid input. Expected JSON with start_date, end_date, and chart_type (pie or bar)."

        start = params.get("start_date")
        end = params.get("end_date")
        chart_type = str(params.get("chart_type", "pie")).lower()

        filters = []
        if start:
            filters.append(f"(Date,ge,exactDate,{start})")
        if end:
            filters.append(f"(Date,lt,exactDate,{end})")

        expenses = expenses_model.list(
            json.dumps({"filters": filters}) if filters else None
        )
        if not expenses:
            return "No expenses found for that period — nothing to chart."

        grand_total = sum(float(e.get("Amount") or 0) for e in expenses)
        period_label = f"{start} to {end}" if start and end else "All Time"
        chart_label = "Daily Spending" if chart_type == "bar" else "Spending Breakdown"
        output_path = f"/tmp/chart_{uuid.uuid4().hex}.png"

        chart_data = [
            {"Date": e.get("Date"), "Item": e.get("Item"), "Amount": float(e.get("Amount") or 0)}
            for e in expenses
        ]

        claude_prompt = (
            f"You are a data visualization expert. Write a complete, self-contained Python script that:\n"
            f"1. Creates a beautiful {'bar chart of daily spending totals' if chart_type == 'bar' else 'pie chart of spending by category (Item field)'} using plotly\n"
            f"2. Uses only the hardcoded data below — do NOT read from files\n"
            f"3. Saves the chart as PNG to exactly: {output_path}\n"
            f"4. Uses template='plotly_dark' and a clean professional style\n"
            f"5. Title: '{chart_label} — {period_label}', subtitle annotation: 'Total: Rs.{grand_total:,.0f}'\n"
            f"6. Calls ONLY fig.write_image('{output_path}') — never fig.show()\n\n"
            f"Data (JSON list of {{Date, Item, Amount}}):\n{json.dumps(chart_data)}\n\n"
            f"Output ONLY the Python script. No explanation. No markdown fences."
        )

        try:
            script_raw = await _run_claude(claude_prompt, user_id=0, timeout=60)
            script = re.sub(r"^```(?:python)?\s*", "", script_raw.strip(), flags=re.MULTILINE)
            script = re.sub(r"\s*```$", "", script.strip(), flags=re.MULTILINE)

            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                raise RuntimeError(stderr.decode())
            if not os.path.exists(output_path):
                raise FileNotFoundError("Chart file not generated.")

            with open(output_path, "rb") as f:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=f,
                    caption=f"📊 *{chart_label}*\nPeriod: {period_label}\nTotal: ₹{grand_total:,.0f}",
                    parse_mode="markdown",
                )
            return f"Chart sent. {len(expenses)} transactions totalling ₹{grand_total:,.0f} for {period_label}."

        except Exception as e:
            logger.error("Claude chart failed, falling back to matplotlib: %s", e, exc_info=True)
            try:
                if chart_type == "bar":
                    buf = spending_bar_chart(expenses, f"Daily Spending — {period_label}")
                    caption = f"📊 *Daily spending trend*\nPeriod: {period_label}\nTotal: ₹{grand_total:,.0f}"
                else:
                    buf = spending_pie_chart(expenses, f"Spending Breakdown — {period_label}")
                    caption = f"🥧 *Category breakdown*\nPeriod: {period_label}\nTotal: ₹{grand_total:,.0f}"
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=buf,
                    caption=caption,
                    parse_mode="markdown",
                )
                return f"Chart sent. {len(expenses)} transactions totalling ₹{grand_total:,.0f} for {period_label}."
            except Exception as e2:
                logger.error("Fallback chart also failed: %s", e2, exc_info=True)
                return f"Failed to generate chart: {e}"
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    return Tool(
        name="Expense_Analytics_Tool",
        description=(
            "Generate visual expense charts (pie or bar). Use when the user asks for charts, "
            "spending breakdown, category analysis, or daily trends. "
            "Input: JSON string with 'start_date' (YYYY-MM-DD), 'end_date' (YYYY-MM-DD, exclusive), "
            "and 'chart_type' ('pie' or 'bar')."
        ),
        func=analytics_tool,
        coroutine=analytics_tool,
    )


@check_authorization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("/start from user %s", update.effective_user.id)
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")


@check_authorization
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in _research_input_queues:
        _research_input_queues[user_id].put(update.message.text.strip())
        await update.message.reply_text("✅ Got it — continuing research...")
        return
    await agent_engage(update, context, update.message.text.strip())


async def agent_engage(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    tools = _static_tools + [
        await _make_wolfram_tool(update, context),
        await _make_analytics_tool(update, context),
    ]

    def _state_modifier(state):
        today = datetime.now().strftime("%Y-%m-%d %A")
        content = "\n".join(SYSTEM_PROMPT) + f"\nToday's date is {today}."
        return [SystemMessage(content=content)] + state["messages"]

    agent = create_react_agent(llm, tools, prompt=_state_modifier, checkpointer=_main_memory)

    logger.info("Chat from user %s: %s", update.effective_user.id, text)
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": f"user_{update.effective_user.id}"}},
        )
        response_text = result["messages"][-1].content
        logger.info("Agent response: %s", response_text)

        if "HERE_IS_IMAGE" in response_text:
            try:
                images_data = response_text.split("\n")
                for image in images_data[1:]:
                    title, image_url = image.split("=>")
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id, photo=image_url, caption=title
                    )
            except Exception as e:
                logger.error("Error sending image: %s", e)
                await update.message.reply_text(f"Error generating image: {e}")
        else:
            await _send_long(update.message.reply_text, response_text, parse_mode="markdown")
    except Exception as e:
        logger.error("Error in chat handler: %s", e)
        await update.message.reply_text(f"An error occurred: {e}")


@check_authorization
async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice_msg = update.message.voice
    new_file  = await context.bot.get_file(voice_msg.file_id)
    file_path = f"{voice_msg.file_id}.ogg"
    await new_file.download_to_drive(file_path)
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai_model.audio.transcriptions.create(
                model=TEXT_TO_SPEECH_MODEL, file=audio_file
            )
        await agent_engage(update, context, transcript.text.strip())
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@check_authorization
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo     = update.message.photo[-1]
    caption   = update.message.caption or ""
    file_info = await context.bot.get_file(photo)
    file_name = os.path.basename(file_info.file_path) if file_info.file_path else f"{photo.file_id}.jpg"
    await file_info.download_to_drive(file_name)
    try:
        with open(file_name, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")
        caption_part = (
            {"type": "input_text", "text": f"The caption of the Image is: {caption} so assume the caption is the <item> for this transaction"}
            if caption
            else {"type": "input_text", "text": "No caption provided."}
        )
        response = openai_model.responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "Summarize the image content in a single sentence.\n"
                        "Except if it is a transaction.\n"
                        "If it is a transaction, Provide response as below message.\n"
                        "Spent <amount> on <item|whoever the money was sent to> on <date>.\n"
                        "If date is not present, assume today's date.\n"
                        "If amount is not present, assume it is zero.\n"
                        "If item is not present, assume it is miscellaneous."
                    )},
                    caption_part,
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }],
        )
        await agent_engage(update, context, response.output_text)
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)


def _mangle_ipv6(ip: str) -> str:
    last = ip[-1]
    mangled = format((int(last, 16) + 3) % 16, "x")
    return ip[:-1] + mangled


def _noip_set(ip: str) -> tuple[str, str]:
    resp = requests.get(
        "https://dynupdate.no-ip.com/nic/update",
        params={"hostname": NOIP_HOSTNAME, "myip": ip},
        auth=(NOIP_USERNAME, NOIP_PASSWORD),
        headers={"User-Agent": "JARVISBot/1.0 " + NOIP_USERNAME},
        timeout=10,
    )
    resp.raise_for_status()
    return ip, resp.text.strip()


@check_authorization
async def ddns(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    subcommand = (_context.args[0].lower() if _context.args else "")
    logger.info("ddns %s from user %s", subcommand, update.effective_user.id)

    if subcommand == "update":
        def _do_update() -> tuple[str, str]:
            ip = requests.get("http://ip1.dynupdate6.no-ip.com/", timeout=10).text.strip()
            return _noip_set(ip)
        try:
            ip, status = await asyncio.to_thread(_do_update)
            if status.startswith(("good", "nochg")):
                await update.message.reply_text(
                    f"✅ DDNS updated.\nIP: `{ip}`\nStatus: `{status}`",
                    parse_mode="markdown",
                )
            else:
                await update.message.reply_text(f"❌ DDNS update failed: `{status}`", parse_mode="markdown")
        except Exception as e:
            logger.error("Error in ddns update: %s", e)
            await update.message.reply_text("❌ DDNS update failed.")

    elif subcommand == "block":
        def _do_block() -> tuple[str, str]:
            real_ip = requests.get("http://ip1.dynupdate6.no-ip.com/", timeout=10).text.strip()
            return _noip_set(_mangle_ipv6(real_ip))
        try:
            ip, status = await asyncio.to_thread(_do_block)
            if status.startswith(("good", "nochg")):
                await update.message.reply_text(
                    f"🔒 DDNS blocked.\nHostname `{NOIP_HOSTNAME}` now points to `{ip}`.\nStatus: `{status}`",
                    parse_mode="markdown",
                )
            else:
                await update.message.reply_text(f"❌ DDNS block failed: `{status}`", parse_mode="markdown")
        except Exception as e:
            logger.error("Error in ddns block: %s", e)
            await update.message.reply_text("❌ DDNS block failed.")

    else:
        await update.message.reply_text(
            "Usage:\n`/ddns update` — set hostname to your current public IP\n`/ddns block` — point hostname to loopback to cut external access",
            parse_mode="markdown",
        )


@check_authorization
async def wakeUpThePC(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("wakeUpThePC from user %s", update.effective_user.id)
    try:
        resp = requests.get(WAKE_RELAY_URL, timeout=5)
        resp.raise_for_status()
        await update.message.reply_text("🔌 Magic packet sent to wake up the PC.")
    except requests.RequestException as e:
        logger.error("Error sending magic packet via relay: %s", e)
        await update.message.reply_text(f"Failed to wake PC: {e}")


@check_authorization
async def genPass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("GenPass from user %s", update.effective_user.id)
    try:
        translation = str.maketrans("/=+-", "abcd")
        num_chars   = int(context.args[0]) if context.args else 13
        password    = ""
        for _ in range(num_chars // 4):
            password += f"{base64.b64encode(ssl.RAND_bytes(13)).decode('utf-8')[:4].translate(translation)}-"
        password = password.strip("-")[:num_chars]
        await update.message.reply_text("🔑 Password Generated:")
        await update.message.reply_text(password)
    except Exception as e:
        logger.error("Unable to generate password: %s", e)
        await update.message.reply_text(f"Failed to generate password: {e}")


async def send_mag_message(bot: telegram.Bot):
    try:
        mag_info_list = mag_manager.mag_model.list(
            json.dumps({"filters": [f"(Date,eq,exactDate,{datetime.now().strftime('%Y-%m-%d')})"]}))
        if not mag_info_list:
            return
        mag_info = mag_info_list[0]
        response = (
            "Todays's MAG:\n"
            f"**📅 Date:** {mag_info['Date']}\n"
            f"**🌖 Tithi:** {mag_info['Tithi']}\n"
        )
        if mag_info.get("Note"):
            response += f"**📝 Note:** {mag_info['Note']}\n"
        await bot.send_message(chat_id=CHAT_ID, text=response, parse_mode="markdown")
        logger.info("Scheduled MAG message sent.")
    except Exception as e:
        logger.error("Error sending scheduled MAG message: %s", e)


@check_authorization
async def get_cmp_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = portfolio_manager.update_cmp()
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="markdown")
    except Exception as e:
        logger.error("Error in get_cmp_today: %s", e)


@check_authorization
async def claude_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Usage: /claudeApi <your prompt>")
        return
    logger.info("claudeApi from user %s: %s", update.effective_user.id, prompt)
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(update.effective_chat.id, context.bot, stop_typing)
    )
    try:
        output = await _run_claude(prompt, update.effective_user.id, timeout=120)
    except Exception as e:
        logger.error("Error in claude_api: %s", e)
        output = f"❌ Error running Claude: {e}"
    finally:
        stop_typing.set()
        typing_task.cancel()
    await _send_long(update.message.reply_text, output, parse_mode="markdown")


@check_authorization
async def portfolio_suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Starting deep portfolio research... this may take a few minutes.")

    try:
        with open(GRAHAM_PROMPT_PATH, "r", encoding="utf-8") as f:
            graham_prompt = f.read()
    except FileNotFoundError:
        await update.message.reply_text(f"❌ GrahamPrompt.md not found at {GRAHAM_PROMPT_PATH}.")
        return

    transactions = portfolio_transactions_model.list()
    if not transactions:
        await update.message.reply_text("❌ No portfolio transactions found.")
        return

    csv_lines = ["Portfolio,Ticker,TransactionType,NoOfShares,CostPerShare,Date,CMP"]
    for tx in transactions:
        csv_lines.append(
            f"{tx.get('Portfolio','')},{tx.get('Ticker','')},{tx.get('TransactionType','')},"
            f"{tx.get('NoOfShares','')},{tx.get('CostPerShare','')},{tx.get('Date','')},{tx.get('CMP','')}"
        )

    user_prompt = (
        f"# My Portfolio Transactions\n\n"
        f"```csv\n{chr(10).join(csv_lines)}\n```\n\n"
        "Search the web for current market prices, latest earnings, analyst ratings, and recent news "
        "for each holding. Perform multiple searches per ticker. Generate the full research report."
    )

    user_id = update.effective_user.id
    _research_input_queues[user_id] = _queue_mod.Queue()

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(update.effective_chat.id, context.bot, stop_typing)
    )
    output, citations, usage = "", [], {}
    try:
        output, citations, usage = await _run_portfolio_research(
            user_prompt,
            system_prompt=graham_prompt,
            timeout=1800,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            user_id=user_id,
        )
    except asyncio.TimeoutError:
        await update.message.reply_text("⏱️ Research timed out. Try again.")
        return
    except Exception as e:
        logger.error("Error in portfolio_suggest: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error: {e}")
        return
    finally:
        _research_input_queues.pop(user_id, None)
        stop_typing.set()
        typing_task.cancel()

    # Send research text first so user can review
    await _send_long(update.message.reply_text, output)

    # Store for PDF callback and ask for confirmation
    context.bot_data[f"research_{user_id}"] = (output, citations, usage)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📄 Generate PDF", callback_data=f"portfolio_pdf_yes_{user_id}"),
        InlineKeyboardButton("❌ Skip", callback_data=f"portfolio_pdf_no_{user_id}"),
    ]])
    await update.message.reply_text(
        "Research complete. Would you like a PDF report?",
        reply_markup=keyboard,
    )


async def portfolio_pdf_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data  # e.g. "portfolio_pdf_yes_12345"

    if "_no_" in data:
        await query.edit_message_text("PDF skipped.")
        return

    user_id = int(data.split("_")[-1])
    stored = context.bot_data.pop(f"research_{user_id}", None)
    if not stored:
        await query.edit_message_text("Session expired. Run /portfolioSuggest again.")
        return

    await query.edit_message_text("⏳ Generating PDF...")
    output, citations, usage = stored
    try:
        pdf_bytes = _report_to_pdf(output, citations=citations, usage=usage)
        filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=io.BytesIO(pdf_bytes),
            filename=filename,
            caption=f"📊 Portfolio Research Report — {usage.get('total_tokens', 0):,} tokens used",
        )
    except Exception as e:
        logger.error("PDF generation failed: %s", e, exc_info=True)
        await context.bot.send_message(chat_id=query.message.chat_id, text=f"❌ PDF failed: {e}")


@check_authorization
async def get_profit_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
        message = portfolio_manager.get_profit_loss_report()
        await _send_long(update.message.reply_text, message, parse_mode="Markdown")
        logger.info("P&L summary sent.")
    except Exception as e:
        logger.error("Error in get_profit_loss: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error generating P&L report: {e}")


async def setup_scheduler(application):
    async def scheduled_update_cmp():
        try:
            msg = portfolio_manager.update_cmp()
            await application.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="markdown")
        except Exception as e:
            logger.error("Error in scheduled CMP update: %s", e)

    scheduler.add_job(
        send_mag_message,
        CronTrigger(hour="8", minute="0"),
        args=[application.bot],
    )
    scheduler.add_job(
        scheduled_update_cmp,
        CronTrigger(hour="8", minute="15", day_of_week="0-4"),
    )
    scheduler.start()
    logger.info("🕒 Scheduler started.")


if __name__ == "__main__":
    logger.info("Starting the Telegram bot application.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(setup_scheduler).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, voice))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, image))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("wakeTheBeast", wakeUpThePC))
    app.add_handler(CommandHandler("genPass", genPass, has_args=1))
    app.add_handler(CommandHandler("ddns", ddns))
    app.add_handler(CommandHandler("updateTicker", get_cmp_today))
    app.add_handler(CommandHandler("getProfitLoss", get_profit_loss))
    app.add_handler(CommandHandler("claudeApi", claude_api))
    app.add_handler(CommandHandler("portfolioSuggest", portfolio_suggest))
    app.add_handler(CallbackQueryHandler(portfolio_pdf_callback, pattern=r"^portfolio_pdf_"))

    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
