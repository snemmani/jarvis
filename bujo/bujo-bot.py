import os
import sys

# Ensure the project root is on sys.path so `bujo.*` imports work when this
# script is run directly (VS Code / debugpy adds the script dir, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import json
import logging
import re
import ssl
from datetime import datetime, timedelta

import telegram
import wolframalpha
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
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
    price_alerts_model,
    RELAY_BASE_URL,
)
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from bujo.portoflio.manage import PortfolioManager
from bujo.portoflio.alerts import run_portfolio_alerts
from bujo.portoflio.rebalance import run_rebalance_analysis, run_fresh_portfolio_analysis
from bujo.analytics.charts import spending_pie_chart, spending_bar_chart
from langchain_openai import ChatOpenAI
import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",
    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",
    "1. Expenses Tool – If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool – If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Portfolio_Tool – If I talk about portfolio transactions (buying or selling stocks/shares, recording a trade, listing my portfolio transactions, depositing or withdrawing cash from a portfolio), use this tool.",
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

_TG_MAX = 4096


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

# user_id -> alert_id awaiting a new target price (modify flow)
_alert_modify_pending: dict[int, int] = {}

# ── Build Portfolio conversation states ──
BP_AMOUNT, BP_RISK, BP_HORIZON, BP_SECTOR_FOCUS, BP_SECTOR_AVOID, BP_STOCK_COUNT, BP_CONFIRM = range(7)

_BP_SECTORS = ["IT / Tech", "Banking / Finance", "Pharma", "FMCG / Consumer", "Manufacturing", "Energy"]


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
        description="Use this tool to record or list portfolio transactions (buy/sell stocks, deposit/withdraw cash).",
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
        except Exception as e:
            logger.error("Chart generation failed: %s", e, exc_info=True)
            return f"Failed to generate chart: {e}"

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
    text = update.message.text.strip()

    # Intercept pending alert modify flow
    if user_id in _alert_modify_pending:
        alert_id = _alert_modify_pending.pop(user_id)
        try:
            new_price = float(text.replace(",", "").replace("₹", "").strip())
        except ValueError:
            await update.message.reply_text("❌ Invalid price. Send just a number, e.g. `1250.50`.")
            _alert_modify_pending[user_id] = alert_id  # put it back
            return
        ok = price_alerts_model.update(alert_id, TargetPrice=new_price)
        if ok:
            await update.message.reply_text(f"✅ Alert updated — new target ₹{new_price:,.2f}")
        else:
            await update.message.reply_text("❌ Failed to update alert.")
        return

    await agent_engage(update, context, text)


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


@check_authorization
async def ddns(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    subcommand = (_context.args[0].lower() if _context.args else "")
    logger.info("ddns %s from user %s", subcommand, update.effective_user.id)

    if subcommand == "update":
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{RELAY_BASE_URL}/ddns/update", timeout=15
            )
            resp.raise_for_status()
            ip, status = resp.text.strip().split(" ", 1)
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
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{RELAY_BASE_URL}/ddns/block", timeout=15
            )
            resp.raise_for_status()
            ip, status = resp.text.strip().split(" ", 1)
            if status.startswith(("good", "nochg")):
                await update.message.reply_text(
                    f"🔒 DDNS blocked.\nHostname now points to `{ip}`.\nStatus: `{status}`",
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
async def get_profit_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
        message = portfolio_manager.get_profit_loss_report()
        await _send_long(update.message.reply_text, message, parse_mode="Markdown")
        logger.info("P&L summary sent.")
    except Exception as e:
        logger.error("Error in get_profit_loss: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error generating P&L report: {e}")


@check_authorization
async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Usage: /setAlert <ticker> <above|below|both> <price> [action notes]"""
    args = context.args or []
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: `/setAlert <ticker> <above|below|both> <price> [action notes]`\n"
            "Example: `/setAlert INFY.NS above 1800 Sell 50 shares`",
            parse_mode="Markdown",
        )
        return
    ticker, direction, price_str = args[0].upper(), args[1].lower(), args[2]
    action = " ".join(args[3:])
    if direction not in ("above", "below", "both"):
        await update.message.reply_text("Direction must be `above`, `below`, or `both`.", parse_mode="Markdown")
        return
    try:
        target_price = float(price_str)
    except ValueError:
        await update.message.reply_text("Price must be a number.")
        return
    result = price_alerts_model.create(ticker, direction, target_price, action)
    if result:
        action_line = f"\n📝 _{action}_" if action else ""
        await update.message.reply_text(
            f"✅ Alert set: *{ticker}* {direction} ₹{target_price:,.2f}{action_line}",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("❌ Failed to create alert. Try again.")


@check_authorization
async def list_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    alerts = price_alerts_model.list_active()
    if not alerts:
        await update.message.reply_text("No active price alerts.")
        return
    for alert in alerts:
        alert_id = alert.get("Id")
        ticker = alert.get("Ticker", "?")
        direction = alert.get("Direction", "?")
        target = alert.get("TargetPrice", 0)
        action = (alert.get("Action") or "").strip()
        action_line = f"\n📝 _{action}_" if action else ""
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✏️ Modify", callback_data=f"pal_modify_{alert_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"pal_cancel_{alert_id}"),
            InlineKeyboardButton("✅ Done", callback_data=f"pal_done_{alert_id}"),
        ]])
        await update.message.reply_text(
            f"🔔 *{ticker}* — {direction} ₹{target:,.2f}{action_line}",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )


async def price_alert_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    async def _try_delete():
        try:
            await query.message.delete()
        except Exception:
            pass  # message too old to delete — ignore

    if data.startswith("pal_done_"):
        await _try_delete()

    elif data.startswith("pal_modify_"):
        alert_id = int(data.split("_")[-1])
        user_id = query.from_user.id
        _alert_modify_pending[user_id] = alert_id
        await _try_delete()
        await query.message.reply_text(
            "✏️ Send the new target price (just a number, e.g. `1250.50`):",
            parse_mode="Markdown",
        )

    elif data.startswith("pal_cancel_"):
        alert_id = int(data.split("_")[-1])
        ok = price_alerts_model.deactivate(alert_id)
        _alert_modify_pending.pop(query.from_user.id, None)
        await _try_delete()
        if not ok:
            await query.message.reply_text("❌ Failed to cancel alert.")


@check_authorization
async def portfolio_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("portfolioAlerts from user %s", update.effective_user.id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        msg = await asyncio.to_thread(run_portfolio_alerts, portfolio_transactions_model)
        if msg:
            await _send_long(update.message.reply_text, msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("✅ No alerts — all positions look clean.")
    except Exception as e:
        logger.error("Error in portfolio_alerts: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error running alerts: {e}")


@check_authorization
async def rebalance_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("rebalanceRecommendations from user %s", update.effective_user.id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    await update.message.reply_text(
        "⏳ Running full portfolio rebalance analysis… this may take a minute."
    )
    try:
        report_path, input_path, usage_msg = await asyncio.to_thread(
            run_rebalance_analysis, portfolio_transactions_model
        )
        for path, caption in [
            (input_path,  "📋 Input prompt — portfolio data + market context for fact-checking"),
            (report_path, "📄 Rebalance report — open in any Markdown viewer"),
        ]:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    await context.bot.send_document(
                        chat_id=update.effective_chat.id,
                        document=f,
                        filename=os.path.basename(path),
                        caption=caption,
                    )
                os.remove(path)
        await update.message.reply_text(usage_msg)
    except Exception as e:
        logger.error("Error in rebalanceRecommendations: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error running rebalance analysis: {e}")


@check_authorization
async def bp_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /buildPortfolio conversation."""
    logger.info("buildPortfolio started by user %s", update.effective_user.id)
    context.user_data.clear()
    # If amount supplied inline, skip directly to risk question
    args = context.args
    if args:
        try:
            amount = float(args[0].replace(",", "").replace("₹", ""))
            context.user_data["bp_amount"] = amount
            return await _bp_ask_risk(update, context)
        except ValueError:
            pass
    await update.message.reply_text(
        "💼 *Build a Fresh Portfolio*\n\nStep 1/6 — How much do you want to invest?\n"
        "Send the amount in ₹ (e.g. `500000` or `5,00,000`)",
        parse_mode="Markdown",
    )
    return BP_AMOUNT


async def bp_got_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    text = update.message.text.strip().replace(",", "").replace("₹", "")
    try:
        amount = float(text)
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Please send a valid positive number, e.g. `500000`.")
        return BP_AMOUNT
    context.user_data["bp_amount"] = amount
    return await _bp_ask_risk(update, context)


async def _bp_ask_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🛡️ Conservative", callback_data="bp_risk_Conservative"),
        InlineKeyboardButton("⚖️ Moderate",     callback_data="bp_risk_Moderate"),
        InlineKeyboardButton("🚀 Aggressive",   callback_data="bp_risk_Aggressive"),
    ]])
    amount = context.user_data["bp_amount"]
    msg = f"✅ Amount: ₹{amount:,.0f}\n\nStep 2/6 — What's your risk appetite?"
    if update.callback_query:
        await update.callback_query.message.reply_text(msg, reply_markup=keyboard)
    else:
        await update.message.reply_text(msg, reply_markup=keyboard)
    return BP_RISK


async def bp_got_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    context.user_data["bp_risk"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📅 Short  <2yr",   callback_data="bp_horizon_<2 years"),
        InlineKeyboardButton("📆 Medium 2–5yr", callback_data="bp_horizon_2–5 years"),
        InlineKeyboardButton("🗓️ Long  >5yr",   callback_data="bp_horizon_Long >5yr"),
    ]])
    await query.message.reply_text(
        f"✅ Risk: {context.user_data['bp_risk']}\n\nStep 3/6 — Investment horizon?",
        reply_markup=keyboard,
    )
    return BP_HORIZON


async def bp_got_horizon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    context.user_data["bp_horizon"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    buttons = [[InlineKeyboardButton("🌐 All sectors", callback_data="bp_focus_All sectors")]]
    buttons += [[InlineKeyboardButton(s, callback_data=f"bp_focus_{s}")] for s in _BP_SECTORS]
    await query.message.reply_text(
        f"✅ Horizon: {context.user_data['bp_horizon']}\n\nStep 4/6 — Any sector you want to *focus* on?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return BP_SECTOR_FOCUS


async def bp_got_focus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    context.user_data["bp_focus"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    buttons = [[InlineKeyboardButton("🚫 None", callback_data="bp_avoid_None")]]
    buttons += [[InlineKeyboardButton(s, callback_data=f"bp_avoid_{s}")] for s in _BP_SECTORS]
    await query.message.reply_text(
        f"✅ Focus: {context.user_data['bp_focus']}\n\nStep 5/6 — Any sector to *avoid*?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return BP_SECTOR_AVOID


async def bp_got_avoid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    context.user_data["bp_avoid"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("5–8 (concentrated)",  callback_data="bp_count_5–8 (concentrated)"),
        InlineKeyboardButton("8–12 (balanced)",     callback_data="bp_count_8–12 (balanced)"),
    ], [
        InlineKeyboardButton("12–15 (diversified)", callback_data="bp_count_12–15 (diversified)"),
        InlineKeyboardButton("🤖 Auto",             callback_data="bp_count_Auto"),
    ]])
    await query.message.reply_text(
        f"✅ Avoid: {context.user_data['bp_avoid']}\n\nStep 6/6 — How many stocks?",
        reply_markup=keyboard,
    )
    return BP_STOCK_COUNT


async def bp_got_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    context.user_data["bp_count"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    d = context.user_data
    summary = (
        f"📋 *Portfolio Build Summary*\n\n"
        f"💰 Amount:  ₹{d['bp_amount']:,.0f}\n"
        f"⚖️ Risk:    {d['bp_risk']}\n"
        f"🗓️ Horizon: {d['bp_horizon']}\n"
        f"🎯 Focus:   {d['bp_focus']}\n"
        f"🚫 Avoid:   {d['bp_avoid']}\n"
        f"📊 Stocks:  {d['bp_count']}\n\n"
        "Shall I build this portfolio? (takes ~2–3 minutes)"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Build it", callback_data="bp_confirm_yes"),
        InlineKeyboardButton("❌ Cancel",   callback_data="bp_confirm_no"),
    ]])
    await query.message.reply_text(summary, parse_mode="Markdown", reply_markup=keyboard)
    return BP_CONFIRM


async def bp_confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return ConversationHandler.END
    query = update.callback_query
    await query.answer()
    await query.message.delete()
    if query.data == "bp_confirm_no":
        await query.message.reply_text("❌ Portfolio build cancelled.")
        context.user_data.clear()
        return ConversationHandler.END

    d = context.user_data
    amount = d["bp_amount"]
    preferences = {
        "risk_appetite":  d.get("bp_risk", "Moderate"),
        "horizon":        d.get("bp_horizon", "Long >5yr"),
        "sector_focus":   d.get("bp_focus", "All sectors"),
        "sector_avoid":   d.get("bp_avoid", "None"),
        "stock_count":    d.get("bp_count", "Auto"),
    }
    context.user_data.clear()

    await query.message.reply_text(
        f"⏳ Screening NSE large & upper mid-cap stocks and building your ₹{amount:,.0f} portfolio…\n"
        "This may take 2–3 minutes.",
    )
    try:
        report_path, input_path, usage_msg = await asyncio.to_thread(
            run_fresh_portfolio_analysis, amount, preferences
        )
        for path, caption in [
            (input_path,  "📋 Input prompt — screened candidates + fundamentals"),
            (report_path, "📄 Fresh portfolio report — open in any Markdown viewer"),
        ]:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    await context.bot.send_document(
                        chat_id=query.message.chat_id,
                        document=f,
                        filename=os.path.basename(path),
                        caption=caption,
                    )
                os.remove(path)
        await query.message.reply_text(usage_msg)
    except Exception as e:
        logger.error("Error in buildPortfolio: %s", e, exc_info=True)
        await query.message.reply_text(f"❌ Error building portfolio: {e}")
    return ConversationHandler.END


async def bp_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ Portfolio build cancelled.")
    return ConversationHandler.END


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

    async def scheduled_portfolio_alerts():
        try:
            await asyncio.to_thread(portfolio_manager.update_cmp)
            msg = await asyncio.to_thread(run_portfolio_alerts, portfolio_transactions_model)
            if msg:
                await application.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="markdown")
        except Exception as e:
            logger.error("Error in scheduled portfolio alerts: %s", e)

    scheduler.add_job(
        scheduled_portfolio_alerts,
        CronTrigger(hour="9", minute="0"),
    )

    async def scheduled_price_alerts():
        try:
            alerts = price_alerts_model.list_active()
            now = datetime.now()
            for alert in alerts:
                alert_id = alert.get("Id")
                ticker = alert.get("Ticker", "")
                direction = alert.get("Direction", "")
                target = float(alert.get("TargetPrice") or 0)


                cmp = yf.Ticker(ticker).info.get("currentPrice") or 0
                if not cmp:
                    continue

                triggered = (
                    (direction == "above" and cmp >= target) or
                    (direction == "below" and cmp <= target) or
                    (direction == "both" and cmp != target)
                )
                if not triggered:
                    continue

                arrow = "📈" if cmp >= target else "📉"
                action = (alert.get("Action") or "").strip()
                action_line = f"\n📝 _{action}_" if action else ""
                await application.bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"🔔 *Price Alert — {ticker}*\n"
                        f"{arrow} CMP ₹{cmp:,.2f} is {direction} target ₹{target:,.2f}"
                        f"{action_line}"
                        f"\n\nUse /listAlerts to modify or cancel."
                    ),
                    parse_mode="Markdown",
                )
        except Exception as e:
            logger.error("Error in scheduled price alerts: %s", e)

    scheduler.add_job(
        scheduled_price_alerts,
        CronTrigger(hour="9,11,13,15", minute="0", day_of_week="0-4"),
    )
    async def scheduled_rebalance():
        try:
            report_path, input_path, usage_msg = await asyncio.to_thread(
                run_rebalance_analysis, portfolio_transactions_model
            )
            for path, caption in [
                (input_path,  "📋 Input prompt — portfolio data + market context for fact-checking"),
                (report_path, "📄 Monthly rebalance report — open in any Markdown viewer"),
            ]:
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        await application.bot.send_document(
                            chat_id=CHAT_ID,
                            document=f,
                            filename=os.path.basename(path),
                            caption=caption,
                        )
                    os.remove(path)
            await application.bot.send_message(chat_id=CHAT_ID, text=usage_msg)
        except Exception as e:
            logger.error("Error in scheduled rebalance: %s", e)

    scheduler.add_job(
        scheduled_rebalance,
        CronTrigger(day=1, hour=9, minute=30),
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
    app.add_handler(CommandHandler("portfolioAlerts", portfolio_alerts))
    app.add_handler(CommandHandler("rebalanceRecommendations", rebalance_recommendations))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("buildPortfolio", bp_start)],
        states={
            BP_AMOUNT:       [MessageHandler(filters.TEXT & ~filters.COMMAND, bp_got_amount)],
            BP_RISK:         [CallbackQueryHandler(bp_got_risk,    pattern=r"^bp_risk_")],
            BP_HORIZON:      [CallbackQueryHandler(bp_got_horizon, pattern=r"^bp_horizon_")],
            BP_SECTOR_FOCUS: [CallbackQueryHandler(bp_got_focus,   pattern=r"^bp_focus_")],
            BP_SECTOR_AVOID: [CallbackQueryHandler(bp_got_avoid,   pattern=r"^bp_avoid_")],
            BP_STOCK_COUNT:  [CallbackQueryHandler(bp_got_count,   pattern=r"^bp_count_")],
            BP_CONFIRM:      [CallbackQueryHandler(bp_confirmed,   pattern=r"^bp_confirm_")],
        },
        fallbacks=[CommandHandler("cancel", bp_cancel)],
        per_message=False,
    ))
    app.add_handler(CommandHandler("setAlert", set_alert))
    app.add_handler(CommandHandler("listAlerts", list_alerts))
    app.add_handler(CallbackQueryHandler(price_alert_callback, pattern=r"^pal_"))
    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
