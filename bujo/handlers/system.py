import asyncio
import base64
import json
import logging
import ssl
from datetime import datetime

import requests
import telegram
from telegram import Update
from telegram.ext import ContextTypes

from bujo.base import check_authorization, WAKE_RELAY_URL, RELAY_BASE_URL, CHAT_ID
from bujo.managers import mag_manager

logger = logging.getLogger(__name__)


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


@check_authorization
async def ddns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subcommand = (context.args[0].lower() if context.args else "")
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
