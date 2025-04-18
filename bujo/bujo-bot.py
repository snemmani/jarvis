from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from bujo.base import ALLOWED_USERS, TELEGRAM_TOKEN, expenses_model
from bujo.expenses.manage import ADD_EXPENSE_CHAT, LIST_EXPENSE_CHAT, ExpenseManager
import requests


# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        await update.message.reply_text("🚫 Sorry, you're not authorized to use this bot.")
        return
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")

async def reveal_my_ipv6(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        await update.message.reply_text("🚫 Sorry, you're not authorized to use this bot.")
        return
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        if response.status_code != 200:
            await update.message.reply_text("Error fetching IPv6 address.")
            return
        ipv6_address = response.json().get("ip", "Unable to fetch IPv6 address")
        await update.message.reply_text(f"Your public IPv6 address is: {ipv6_address}")
        return
    except requests.RequestException as e:
        await update.message.reply_text(f"Error fetching IPv6 address: {e}")

# Main runner
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    expense_manager = ExpenseManager(expenses_model)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("add_expenses", expense_manager.start_add)],
        states={
            ADD_EXPENSE_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, expense_manager.handle_expense_input)]
        },
        fallbacks=[CommandHandler("cancel", expense_manager.cancel)],
    )

    conv_handler_2 = ConversationHandler(
        entry_points=[CommandHandler("list_expenses", expense_manager.start_list)],
        states={
            LIST_EXPENSE_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, expense_manager.handle_list_input)]
        },
        fallbacks=[CommandHandler("cancel", expense_manager.cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ipv6", reveal_my_ipv6))
    app.add_handler(conv_handler)
    app.add_handler(conv_handler_2)

    print("🤖 Bot is running...")
    app.run_polling()
