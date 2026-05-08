import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bujo.base import check_authorization, price_alerts_model

logger = logging.getLogger(__name__)

# user_id -> alert_id awaiting a new target price (modify flow)
_alert_modify_pending: dict[int, int] = {}


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
        ticker    = alert.get("Ticker", "?")
        direction = alert.get("Direction", "?")
        target    = alert.get("TargetPrice", 0)
        action    = (alert.get("Action") or "").strip()
        action_line = f"\n📝 _{action}_" if action else ""
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✏️ Modify", callback_data=f"pal_modify_{alert_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"pal_cancel_{alert_id}"),
            InlineKeyboardButton("✅ Done",   callback_data=f"pal_done_{alert_id}"),
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
            pass

    if data.startswith("pal_done_"):
        await _try_delete()

    elif data.startswith("pal_modify_"):
        alert_id = int(data.split("_")[-1])
        _alert_modify_pending[query.from_user.id] = alert_id
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
