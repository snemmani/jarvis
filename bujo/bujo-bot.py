from re import sub
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from bujo.base import TELEGRAM_TOKEN, expenses_model, mag_model, llm, check_authorization, WOLFRAM_APP_ID, PC_MAC_ADDRESS, BROADCAST_IP, scheduler, CHAT_ID
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
import requests
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from datetime import datetime
import wolframalpha
from wakeonlan import send_magic_packet
import logging
import sys
from typing import List, Dict
import telegram
from apscheduler.triggers.cron import CronTrigger
import json
import ssl
import base64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",

    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",

    "1. Expenses Tool – If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool – If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Wolfram Alpha Tool – If I ask about current events, latest news, mathematical questions, astronomical questions, conversions between units, flight ticket fares, nutrition information of food, stock prices, use this tool.",
    "4. Translation Tool – If I ask for translation:\n   - And I do not specify source/target languages, assume English to Sanskrit.\n   - If I do specify the languages, translate accordingly.",

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
        name="Translation Tool",
        func=lambda x: 'This tool must be awaited',
        description="Use this tool to translate from one language to another.",
        coroutine=lambda x: llm.ainvoke([HumanMessage(x)])
    )
]

# Tool function with runtime-bound user_id
async def make_wolfram_alpha_tool(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tool:
    async def query_tool(query: str):
        logger.info(f"Queriying Wolfram Alpha for query: {query}")
        response = await wolfram_client.aquery(query)
        if hasattr(response, 'pod'):
            for pod in response['pod']:
                subpod = pod['subpod']
                if type(subpod) is list:
                    for item in subpod:
                        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=item['img']['@src'], caption=item['@title'])
                else:
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=subpod['img']['@src'], caption=pod['@title'])
        return "Response completed."
    
    return Tool(
        name="Wolfram Alpha Tool",
        description="Wolfram Alpha Tool for complex or realtime queries.",
        func=query_tool,
        coroutine=query_tool
    )

# Start command
@check_authorization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"/start command received from user: {update.effective_user.id}")
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")

@check_authorization
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Initialize the tool here
    tools_copy = tools.copy()
    tools_copy.append(await make_wolfram_alpha_tool(update, context))
    agent = initialize_agent(
        tools=tools_copy, 
        llm=llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        memory=memory,
        # handle_parsing_errors=True,
        verbose=True
    )
    text = update.message.text.strip()
    logger.info(f"Received chat message from user {update.effective_user.id}: {text}")
    sys_prompt = SYSTEM_PROMPT.copy()
    sys_prompt.append(f'Today\'s date is {datetime.now().strftime("%Y-%m-%d %A")}')
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
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
        

@check_authorization
async def genPass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"GenPass triggered from user: {update.effective_user.id}")
    try:
        translation = str.maketrans('/=+-','abcd')
        num_chars = int(context.args[0]) if context.args else 13
        password = ""
        for iter in range(num_chars//4):
            password += f"{base64.b64encode(ssl.RAND_bytes(13)).decode('utf-8')[:4].translate(translation)}-"
        
        password = password.strip('-')[:num_chars]
        await update.message.reply_text("🔑 Password Generated:")
        await update.message.reply_text(password)
        logger.info("Secure password generated.")
    except requests.RequestException as e:
        logger.error(f"Unable to Generate password: {e}")
        await update.message.reply_text(f"Failed to generate password: {e}")

async def send_mag_message(bot: telegram.Bot):
    """
    Function to send a message to the MAG channel.
    This can be scheduled to run periodically.
    """
    try:
        mag_info_list = mag_manager.mag_model.list(json.dumps({"filters": [f"(Date,eq,exactDate,{datetime.now().strftime('%Y-%m-%d')})"]}))
        if len(mag_info_list) == 0:
            response = "No MAG entries found for today."
            return
        else:
            mag_info = mag_info_list[0]
            response = "Todays's MAG:\n" + \
                f"**📅 Date:** {mag_info['Date']}\n" + \
                f"**🌖 Tithi:** {mag_info['Tithi']}\n" 
            
            if mag_info['Note']:
                response += f"**📝 Note:** {mag_info['Note']}\n"
        # Replace with actual logic to fetch or generate the message
        await bot.send_message(chat_id=CHAT_ID, text=response, parse_mode='markdown')
        logger.info("Scheduled MAG message sent successfully.")
    except Exception as e:
        logger.error(f"Error sending scheduled MAG message: {e}")


async def setup_scheduler(application):
    scheduler.add_job(
        send_mag_message,
        CronTrigger(hour="8", minute="0", day="*", month="*", day_of_week="*"),
        args=[application.bot]
    )
    scheduler.start()
    logger.info("🕒 Scheduler started for sending calendar at 8:00.")

# Main runner
if __name__ == '__main__':
    logger.info("Starting the Telegram bot application.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(setup_scheduler).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("wakeTheBeast", wakeUpThePC))
    app.add_handler(CommandHandler("genPass", genPass, has_args=1))

    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
