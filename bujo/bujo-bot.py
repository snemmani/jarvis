from ast import mod, parse
from types import coroutine
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from bujo.base import ALLOWED_USERS, TELEGRAM_TOKEN, expenses_model, mag_model, llm, check_authorization, SERP_API_KEY, WOLFRAM_APP_ID, PC_MAC_ADDRESS, BROADCAST_IP
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import trim_messages
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from datetime import datetime
import wolframalpha
from wakeonlan import send_magic_packet
from langchain.prompts.chat import ChatPromptTemplate
import logging
import sys
import json
from typing import List, Dict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",

    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",

    "1. Expenses Tool – If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool – If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Web Search Tool – If I ask about news, explanations of concepts, or current information, use this tool. The tool returns a JSON object. You must convert the content into natural language before replying to me.",
    "4. Wolfram Alpha Tool – If I ask about complex mathematics, stock prices, or trends, use this tool for accurate answers.",
    "5. Wolfram Alpha Image Generator – If I ask for visualizations or generated images, use this tool. Do not modify the image URL or output — return it directly as-is.",
    "6. Translation Tool – If I ask for translation:\n   - And I do not specify source/target languages, assume English to Sanskrit.\n   - If I do specify the languages, translate accordingly.",

    "MOST IMPORTANT INSTRUCTIONS:",
    "Always call the appropriate tool based on my latest message. You must never answer directly without invoking a tool first.",
    "When sending a response (either to the LLM for summarization or to me), always return it as a string, not as a JSON object.",
    "Always return tool results in markdown format for readability.",
    "Separate individual items in the tool results with new lines.",
    "Use emojis wherever appropriate to make the response more friendly and visually appealing. 🎯🧮📅💸📰🌐🔢📊🖼️🔤"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def prepend_system_prompt(user_input: str, sys_prompt: str) -> str:
    return f"{sys_prompt}\n\nUser: {user_input}"

def build_chat_messages(user_input: str, sys_prompt: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content':sys_prompt},
        {'role':'human', 'content': user_input}
    ]

expense_manager = ExpenseManager(expenses_model, mag_model)
mag_manager = MagManager(mag_model)
memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)

wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)

async def wolfram_alpha_image_generator(query):
    logger.info(f"Generating image with Wolfram Alpha for query: {query}")
    response = await wolfram_client.aquery(query)
    if hasattr(response, 'pod') and len(response.pod) > 0:
        logger.info("Image generated successfully.")
        response_text = 'HERE_IS_IMAGE:'
        for image in response.pod:
            response_text += f"\n{image.title}=>{image.subpod.img.src}"
        return response_text
    else:
        logger.warning("Failed to generate image.")
        return 'Failed to generate image.'

tools = [
    Tool(
        name="Expenses Interaction",
        func=expense_manager.agent_expenses,
        description="Use this tool to fetch expenses based on specific filters.",
        return_direct=True,

    ),
    Tool(
        name="MAG interaction",
        func=mag_manager.agent_mag,
        description="Use this tool to manage MAG.",
        return_direct=True
    ),
    Tool(
        name="Search the Web",
        func=lambda query: json.dumps(requests.get(f"https://serpapi.com/search", params={"q": query, "api_key": SERP_API_KEY}).json(), indent=2),
        description="Use this tool to search the web for information based on the user's query."
    ),
    Tool(
        name="Wolfram Alpha",
        func=lambda query: next((wolfram_client.query(query)).results).text,
        description="Use this tool for computing and answering complex queries using Wolfram Alpha."
    ),
    Tool(
        name="Wolfram Alpha Image Generator",
        func=lambda x: 'This tool must be awaited',
        description="Use this tool to generate images for queries using Wolfram Alpha.",
        return_direct=True,
        coroutine=wolfram_alpha_image_generator
    ),
    Tool(
        name="Translation Tool",
        func=lambda x: 'This tool must be awaited',
        description="Use this tool to translate from one language to another.",
        coroutine=lambda x: llm.ainvoke([HumanMessage(x)])
    )
]

agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory,
    # handle_parsing_errors=True,
    verbose=True,
    debug=True
)

# Start command
@check_authorization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"/start command received from user: {update.effective_user.id}")
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")

@check_authorization
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    logger.info(f"Received chat message from user {update.effective_user.id}: {text}")
    sys_prompt = SYSTEM_PROMPT.copy()
    sys_prompt.append(f'Today\'s date is {datetime.now().strftime("%Y-%m-%d %A")}')
    try:
        response = await agent.ainvoke(prepend_system_prompt(text, sys_prompt))
        logger.info(f"Agent response: {response}")
        if 'output' in response and "HERE_IS_IMAGE" in response['output']:
            try:
                images_data = response['output'].split('\n')
                for image in images_data[1:]:
                    title, image_url = image.split('=>')
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_url, caption=title)
                logger.info(f"Sent images to user {update.effective_user.id}: {image_url}")
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                await update.message.reply_text(f"Error generating image: {e}")
        else:
            await update.message.reply_text(
                response['output'],
                parse_mode='markdown',
            )
    except Exception as e:
        logger.error(f"Error in chat handler: {e}")
        await update.message.reply_text(f"An error occurred: {e}")

@check_authorization
async def wakeUpThePC(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"wakeUpThePC command received from user: {update.effective_user.id}")
    try:
        send_magic_packet(PC_MAC_ADDRESS, ip_address=BROADCAST_IP)
        await update.message.reply_text("🔌 Magic packet sent to wake up the PC.")
        logger.info("Magic packet sent successfully.")
    except requests.RequestException as e:
        logger.error(f"Error sending magic packet: {e}")
        await update.message.reply_text(f"Waking up the PC: {e}")

# Main runner
if __name__ == '__main__':
    logger.info("Starting the Telegram bot application.")
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
    app.add_handler(CommandHandler("wakeTheBeast", wakeUpThePC))
    # app.add_handler(expense_add_handler)
    # app.add_handler(expenses_list_handler)
    # app.add_handler(modify_mag_handler)
    # app.add_handler(mag_list_handler)

    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
