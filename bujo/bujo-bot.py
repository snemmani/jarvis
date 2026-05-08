import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import base64
import json
import logging
import re
import ssl
from datetime import datetime

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
    llm,
    check_authorization,
    WOLFRAM_APP_ID,
    scheduler,
    CHAT_ID,
    openai_model,
    TEXT_TO_SPEECH_MODEL,
    portfolio_transactions_model,
    price_alerts_model,
)
from bujo.managers import expense_manager, mag_manager, portfolio_manager
from bujo.handlers.utils import send_long
from bujo.handlers.system import send_mag_message, wakeUpThePC, genPass, ddns
from bujo.handlers.alerts import set_alert, list_alerts, price_alert_callback, _alert_modify_pending
from bujo.handlers.portfolio import (
    get_cmp_today, portfolio_alerts, rebalance_recommendations,
    portfolio_dashboard, portfolio_dashboard_callback, rebalance_approval_callback,
    bp_start, bp_got_amount, bp_got_risk, bp_got_horizon, bp_got_focus,
    bp_got_avoid, bp_got_count, bp_confirmed, bp_cancel,
    BP_AMOUNT, BP_RISK, BP_HORIZON, BP_SECTOR_FOCUS, BP_SECTOR_AVOID, BP_STOCK_COUNT, BP_CONFIRM,
)
from bujo.portoflio.alerts import run_portfolio_alerts
from bujo.portoflio.rebalance import run_rebalance_analysis
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


class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except ValueError as e:
            if "I/O operation on closed file" in str(e):
                pass
            else:
                raise


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[SafeStreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)

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

        filter_parts = []
        if start:
            filter_parts.append(f"(Date,ge,exactDate,{start})")
        if end:
            filter_parts.append(f"(Date,lt,exactDate,{end})")

        expenses = expenses_model.list(
            json.dumps({"filters": filter_parts}) if filter_parts else None
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

    if user_id in _alert_modify_pending:
        alert_id = _alert_modify_pending.pop(user_id)
        try:
            new_price = float(text.replace(",", "").replace("₹", "").strip())
        except ValueError:
            await update.message.reply_text("❌ Invalid price. Send just a number, e.g. `1250.50`.")
            _alert_modify_pending[user_id] = alert_id
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
            await send_long(update.message.reply_text, response_text, parse_mode="markdown")
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
            if not alerts:
                return

            # Batch-fetch prices for all alert tickers in one yfinance call
            tickers_list = list({a.get("Ticker", "") for a in alerts if a.get("Ticker")})
            raw = await asyncio.to_thread(
                yf.download, tickers_list, period="1d", progress=False, auto_adjust=True
            )
            close = raw["Close"]
            def get_price(ticker: str) -> float:
                try:
                    col = close if len(tickers_list) == 1 else close[ticker]
                    return float(col.dropna().iloc[-1])
                except Exception:
                    return 0.0

            for alert in alerts:
                ticker    = alert.get("Ticker", "")
                direction = alert.get("Direction", "")
                target    = float(alert.get("TargetPrice") or 0)
                cmp       = get_price(ticker)
                if not cmp:
                    continue

                triggered = (
                    (direction == "above" and cmp >= target) or
                    (direction == "below" and cmp <= target) or
                    (direction == "both"  and cmp != target)
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
    app.add_handler(CommandHandler("start",                   start))
    app.add_handler(CommandHandler("wakeTheBeast",            wakeUpThePC))
    app.add_handler(CommandHandler("genPass",                 genPass, has_args=1))
    app.add_handler(CommandHandler("ddns",                    ddns))
    app.add_handler(CommandHandler("updateTicker",            get_cmp_today))
    app.add_handler(CommandHandler("portfolioDashboard",      portfolio_dashboard))
    app.add_handler(CommandHandler("portfolioAlerts",         portfolio_alerts))
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
    app.add_handler(CommandHandler("setAlert",   set_alert))
    app.add_handler(CommandHandler("listAlerts", list_alerts))
    app.add_handler(CallbackQueryHandler(price_alert_callback, pattern=r"^pal_"))
    app.add_handler(CallbackQueryHandler(rebalance_approval_callback, pattern=r"^rbal_"))
    app.add_handler(CallbackQueryHandler(portfolio_dashboard_callback, pattern=r"^pdash_"))
    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
