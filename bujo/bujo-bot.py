from ast import mod
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from bujo.base import ALLOWED_USERS, TELEGRAM_TOKEN, expenses_model, mag_model, llm, check_authorization, SERP_API_KEY, WOLFRAM_APP_ID
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from langchain.memory import ConversationBufferMemory
import requests
from langchain.agents import initialize_agent, Tool
from datetime import datetime
import wolframalpha

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calender (which I call MAG).",
    "Depending upon my request you will interact with the specific tool, either Expenses tool or MAG tool or Search the Web."
    "If I am talking to you about expenses, you will use the Expenses tool to add or list my expenses.",
    "If I am talking to you about MAG, you will use the MAG tool to add or list my MAG.",
    "If I am talking to you about latest news, or any knowledge article asking you to exlain something, you use the Search the web tool.",
    "If I am talking to you about complex mathematics, prices of stocks or trends of stocks, or for complex tasks that google might not be handled, try wolfram alpha tool."
    "If I am talking to you about generating images or visualizations, use wolfram alpha image generator tool.",
]

def prepend_system_prompt(user_input: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {user_input}"

expense_manager = ExpenseManager(expenses_model, mag_model)
mag_manager = MagManager(mag_model)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)


def wolfram_alpha_image_generator(query):
    response = wolfram_client.query(query)
    if hasattr(response, 'pod') and len(response.pod) > 0 and hasattr(response.pod[0], 'subpod') and len(response.pod[0].subpod) > 0:
        return response.pod[0].subpod[0].img.src
    else:
        return 'Failed to generate image.'

tools = [
    Tool(
        name="Expenses Interaction",
        func=expense_manager.agent_expenses,
        description="Use this tool to fetch expenses based on specific filters."
    ),
    Tool(
        name="MAG interaction",
        func=mag_manager.agent_mag,
        description="Use this tool to manage MAG."
    ),
    Tool(
        name="Search the Web",
        func=lambda query: requests.get(f"https://serpapi.com/search", params={"q": query, "api_key": SERP_API_KEY}).json(),
        description="Use this tool to search the web for information based on the user's query."
    ),
    Tool(
        name="Wolfram Alpha",
        func=lambda query: next(wolfram_client.query(query).results).text,
        description="Use this tool for computing and answering complex queries using Wolfram Alpha."
    ),
    Tool(
        name="Wolfram Alpha Image Generator",
        func=wolfram_alpha_image_generator,
        description="Use this tool to generate images for queries using Wolfram Alpha."
    )
]


agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent="chat-conversational-react-description", 
    memory=memory,
    # handle_parsing_errors=True,
    verbose=True
)



# Start command
@check_authorization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")

@check_authorization
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    SYSTEM_PROMPT.append(f'Today\'s date is {datetime.now().strftime("%Y-%m-%d %A")}')
    response = agent.run(prepend_system_prompt(text))

    if "Wolfram Alpha Image Generator" in response:
        try:
            image_url = response.split("Wolfram Alpha Image Generator: ")[1].strip()
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_url)
        except Exception as e:
            await update.message.reply_text(f"Error generating image: {e}")
    else:
        await update.message.reply_text(
            response
        ) 

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
    

    # expense_add_handler = ConversationHandler(
    #     entry_points=[CommandHandler("add_expenses", expense_manager.start_add)],
    #     states={
    #         ADD_EXPENSE_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, expense_manager.handle_expense_input)]
    #     },
    #     fallbacks=[CommandHandler("cancel", expense_manager.cancel)],
    # )

    # expenses_list_handler = ConversationHandler(
    #     entry_points=[CommandHandler("list_expenses", expense_manager.start_list)],
    #     states={
    #         LIST_EXPENSE_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, expense_manager.handle_list_input)]
    #     },
    #     fallbacks=[CommandHandler("cancel", expense_manager.cancel)],
    # )

    # modify_mag_handler = ConversationHandler(
    #     entry_points=[CommandHandler("update_mag", mag_manager.start_modify)],
    #     states={
    #         UPDATE_MAG: [MessageHandler(filters.TEXT & ~filters.COMMAND, mag_manager.handle_mag_change)]
    #     },
    #     fallbacks=[CommandHandler("cancel", mag_manager.cancel)],
    # )

    # mag_list_handler = ConversationHandler(
    #     entry_points=[CommandHandler("list_mag", mag_manager.start_list)],
    #     states={
    #         LIST_EXPENSE_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, mag_manager.handle_list_input)]
    #     },
    #     fallbacks=[CommandHandler("cancel", expense_manager.cancel)],
    # )

    

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ipv6", reveal_my_ipv6))
    # app.add_handler(expense_add_handler)
    # app.add_handler(expenses_list_handler)
    # app.add_handler(modify_mag_handler)
    # app.add_handler(mag_list_handler)

    print("🤖 Bot is running...")
    app.run_polling()
