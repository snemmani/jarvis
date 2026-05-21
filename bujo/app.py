import logging

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

from bujo.base import TELEGRAM_TOKEN
from bujo.handlers.alerts import list_alerts, price_alert_callback, set_alert
from bujo.handlers.chat import chat, image, start, voice
from bujo.handlers.portfolio import (
    BP_AMOUNT,
    BP_CONFIRM,
    BP_HORIZON,
    BP_RISK,
    BP_SECTOR_AVOID,
    BP_SECTOR_FOCUS,
    BP_STOCK_COUNT,
    bp_cancel,
    bp_confirmed,
    bp_got_amount,
    bp_got_avoid,
    bp_got_count,
    bp_got_focus,
    bp_got_horizon,
    bp_got_risk,
    bp_start,
    fresh_portfolio_approval_callback,
    get_cmp_today,
    portfolio_alerts,
    portfolio_dashboard,
    portfolio_dashboard_callback,
    rebalance_approval_callback,
    rebalance_recommendations,
)
from bujo.handlers.system import ddns, genPass, wakeUpThePC
from bujo.logging_config import configure_logging
from bujo.scheduler import setup_scheduler

logger = logging.getLogger(__name__)


def build_application():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(setup_scheduler).build()

    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("buildPortfolio", bp_start)],
        states={
            BP_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, bp_got_amount)],
            BP_RISK: [CallbackQueryHandler(bp_got_risk, pattern=r"^bp_risk_")],
            BP_HORIZON: [CallbackQueryHandler(bp_got_horizon, pattern=r"^bp_horizon_")],
            BP_SECTOR_FOCUS: [CallbackQueryHandler(bp_got_focus, pattern=r"^bp_focus_")],
            BP_SECTOR_AVOID: [CallbackQueryHandler(bp_got_avoid, pattern=r"^bp_avoid_")],
            BP_STOCK_COUNT: [CallbackQueryHandler(bp_got_count, pattern=r"^bp_count_")],
            BP_CONFIRM: [CallbackQueryHandler(bp_confirmed, pattern=r"^bp_confirm_")],
        },
        fallbacks=[CommandHandler("cancel", bp_cancel)],
        per_message=False,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, voice))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, image))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("wakeTheBeast", wakeUpThePC))
    app.add_handler(CommandHandler("genPass", genPass, has_args=1))
    app.add_handler(CommandHandler("ddns", ddns))
    app.add_handler(CommandHandler("updateTicker", get_cmp_today))
    app.add_handler(CommandHandler("portfolioDashboard", portfolio_dashboard))
    app.add_handler(CommandHandler("portfolioAlerts", portfolio_alerts))
    app.add_handler(CommandHandler("rebalanceRecommendations", rebalance_recommendations))
    app.add_handler(CommandHandler("setAlert", set_alert))
    app.add_handler(CommandHandler("listAlerts", list_alerts))
    app.add_handler(CallbackQueryHandler(price_alert_callback, pattern=r"^pal_"))
    app.add_handler(CallbackQueryHandler(rebalance_approval_callback, pattern=r"^rbal_"))
    app.add_handler(CallbackQueryHandler(fresh_portfolio_approval_callback, pattern=r"^bp_llm_"))
    app.add_handler(CallbackQueryHandler(portfolio_dashboard_callback, pattern=r"^pdash_"))
    return app


def run() -> None:
    configure_logging()
    logger.info("Starting the Telegram bot application.")
    app = build_application()
    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
