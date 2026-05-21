import asyncio
import logging
import os

import yfinance as yf
from apscheduler.triggers.cron import CronTrigger

from bujo.base import CHAT_ID, portfolio_transactions_model, price_alerts_model, scheduler
from bujo.handlers.system import send_mag_message
from bujo.managers import portfolio_manager
from bujo.portoflio.alerts import run_portfolio_alerts
from bujo.portoflio.rebalance import run_rebalance_analysis

logger = logging.getLogger(__name__)


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
                ticker = alert.get("Ticker", "")
                direction = alert.get("Direction", "")
                target = float(alert.get("TargetPrice") or 0)
                cmp = get_price(ticker)
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
                (input_path, "📋 Input prompt — portfolio data + market context for fact-checking"),
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
