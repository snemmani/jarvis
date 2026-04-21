import os
import sys

# Ensure the project root is on sys.path so `bujo.*` imports work when this
# script is run directly (VS Code / debugpy adds the script dir, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import logging
import ssl
from datetime import datetime

import telegram
import wolframalpha
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
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
    scheduler,
    CHAT_ID,
    openai_model,
    TEXT_TO_SPEECH_MODEL,
    portfolio_transactions_model,
    NOIP_USERNAME,
    NOIP_PASSWORD,
    NOIP_HOSTNAME,
)
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from bujo.portoflio.manage import PortfolioManager
from bujo.analytics.charts import spending_pie_chart, spending_bar_chart
from wakeonlan import send_magic_packet
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

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
        import json as _json
        logger.info("Expense analytics request: %s", query)
        try:
            params = _json.loads(query.replace("```json", "").replace("```", "").strip())
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
            return (
                f"Chart sent. {len(expenses)} transactions totalling ₹{grand_total:,.0f} "
                f"for the period {period_label}."
            )
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


@check_authorization
async def updateDDNS(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    logger.info("updateDDNS from user %s", update.effective_user.id)
    try:
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            "noip-duc",
            "--username", NOIP_USERNAME,
            "--password", NOIP_PASSWORD,
            "-g", NOIP_HOSTNAME,
            "--ip-method", "http://ip1.dynupdate6.no-ip.com/",
            "--once",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode().strip() or "(no output)"
        if proc.returncode == 0:
            await update.message.reply_text(f"✅ DDNS updated.\n`{output}`", parse_mode="markdown")
        else:
            await update.message.reply_text(f"❌ DDNS update failed (exit {proc.returncode}).\n`{output}`", parse_mode="markdown")
    except FileNotFoundError:
        await update.message.reply_text("❌ `noip-duc` not found. Is it installed?", parse_mode="markdown")
    except Exception as e:
        logger.error("Error in updateDDNS: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


@check_authorization
async def wakeUpThePC(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("wakeUpThePC from user %s", update.effective_user.id)
    try:
        send_magic_packet(PC_MAC_ADDRESS, ip_address=BROADCAST_IP)
        await update.message.reply_text("🔌 Magic packet sent to wake up the PC.")
    except requests.RequestException as e:
        logger.error("Error sending magic packet: %s", e)
        await update.message.reply_text(f"Waking up the PC: {e}")


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
    app.add_handler(CommandHandler("updateDDNS", updateDDNS))
    app.add_handler(CommandHandler("updateTicker", get_cmp_today))
    app.add_handler(CommandHandler("getProfitLoss", get_profit_loss))

    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
