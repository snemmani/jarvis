import base64
import logging
import os

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

from bujo.agent import agent_engage
from bujo.base import (
    OPENAI_MODEL,
    TEXT_TO_SPEECH_MODEL,
    check_authorization,
    openai_model,
    price_alerts_model,
)
from bujo.handlers.alerts import _alert_modify_pending

logger = logging.getLogger(__name__)


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
            return ConversationHandler.END
        ok = price_alerts_model.update(alert_id, TargetPrice=new_price)
        if ok:
            await update.message.reply_text(f"✅ Alert updated — new target ₹{new_price:,.2f}")
        else:
            await update.message.reply_text("❌ Failed to update alert.")
        return ConversationHandler.END

    await agent_engage(update, context, text)


@check_authorization
async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice_msg = update.message.voice
    new_file = await context.bot.get_file(voice_msg.file_id)
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
    photo = update.message.photo[-1]
    caption = update.message.caption or ""
    file_info = await context.bot.get_file(photo)
    file_name = os.path.basename(file_info.file_path) if file_info.file_path else f"{photo.file_id}.jpg"
    await file_info.download_to_drive(file_name)
    try:
        with open(file_name, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")
        caption_instruction = (
            f"USER CAPTION: \"{caption}\"\n\n"
            "The caption is the user's own words and has HIGHER authority than anything visible in the image.\n"
            "Apply it as follows — the caption may contain any combination of these overrides:\n"
            "  • A stock ticker (e.g. INFY.NS, TCS.BO) → use this as Ticker instead of what the image shows.\n"
            "  • A portfolio name (e.g. LT, Core, Default) → use this as Portfolio.\n"
            "  • A note, thesis, condition, or rationale (e.g. 'exit on governance shock', 'trim if NIM < 9%') "
            "→ include this verbatim in the Note field.\n"
            "  • Any combination of the above — apply all corrections simultaneously.\n"
            "Whatever the caption says overrides the image for that field. "
            "The note portion must be copied exactly — do not paraphrase or shorten it."
            if caption
            else "No caption was provided. Use values from the image only."
        )
        response = openai_model.responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "Look at this image and classify it as one of three types:\n\n"
                        "TYPE 1 — EXPENSE (UPI payment, bill, receipt, money transfer):\n"
                        "  Respond: Spent <amount> on <item or recipient> on <date>.\n"
                        "  If no date, use today. If no amount, use zero. If no item, use miscellaneous.\n\n"
                        "TYPE 2 — PORTFOLIO TRANSACTION (stock broker app, trade confirmation, "
                        "buy/sell order, demat statement):\n"
                        "  Respond in this exact format:\n"
                        "  Portfolio transaction: <Buy|Sell> <shares> shares of <TICKER> at ₹<price> on <date>."
                        " Portfolio: <portfolio name or 'Default'>."
                        " Note: <note text, or 'None' if no note>.\n"
                        "  Ticker must include exchange suffix (.NS for NSE, .BO for BSE).\n"
                        "  If no date visible, use today.\n\n"
                        "TYPE 3 — ANYTHING ELSE:\n"
                        "  Respond with a one-sentence summary of the image.\n\n"
                        f"{caption_instruction}"
                    )},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }],
        )
        await agent_engage(update, context, response.output_text)
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)
