import cohere
import os

from langchain_openai import ChatOpenAI
from bujo.models.expenses import Expenses
from bujo.models.mag import MAG
from dotenv import load_dotenv
from functools import wraps
from langchain_cohere import ChatCohere
from telegram.ext import ConversationHandler
from telegram import Update
from telegram.ext import ContextTypes
import logging

#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s [%(levelname)s] %(message)s",
#    handlers=[logging.StreamHandler()]  # sends to stdout/stderr
#)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
# Globals
ALLOWED_USERS = [int(os.environ["TELEGRAM_USER_ID"])]
# COHERE_MODEL = os.environ["COHERE_MODEL"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]
NOCODB_BASE_URL = os.environ["NOCODB_BASE_URL"]

# Secrets
# COHERE_API_KEY = os.environ['COHERE_API_KEY']
NOCODB_API_TOKEN = os.environ['NOCODB_API_TOKEN']
NOCODB_EXPENSES_TABLE_ID = os.environ["NOCODB_EXPENSES_TABLE_ID"]
NOCODB_MAG_TABLE_ID = os.environ["NOCODB_MAG_TABLE_ID"]
NOCODB_EXPENSES_MAG_LINK_ID = os.environ["NOCODB_EXPENSES_MAG_LINK_ID"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
SERP_API_KEY = os.environ["SERP_API_KEY"]
WOLFRAM_APP_ID = os.environ["WOLFRAM_APP_ID"]
PC_MAC_ADDRESS = os.environ["PC_MAC_ADDRESS"]
BROADCAST_IP = os.environ["BROADCAST_IP"]
# Initializations
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
mag_model = MAG(NOCODB_BASE_URL, NOCODB_API_TOKEN, NOCODB_MAG_TABLE_ID)
expenses_model = Expenses(NOCODB_BASE_URL, NOCODB_API_TOKEN, NOCODB_EXPENSES_TABLE_ID, NOCODB_EXPENSES_MAG_LINK_ID, mag_model)
expense_add_messages=[
    {'role': 'system', 'content': 'You are an expert freetext to python serializer'},
    {'role': 'system', 'content': 'There are following fields in expenses schema: date, item, amount'},
    {'role': 'system', 'content': 'Your role is to convert whatever data that I provide into a JSON. You will respond witha  json object/list only. Nothing else.'},
]

def check_authorization(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Support both method and function handlers
            if len(args) < 2:
                raise TypeError("Expected at least 2 arguments (update, context) or (self, update, context)")

            if hasattr(args[0], 'effective_user'):  # non-class function
                update = args[0]
                context = args[1]
                user_id = update.effective_user.id
                if user_id not in ALLOWED_USERS:
                    await update.message.reply_text("🚫 You're not authorized to use this bot.")
                    return ConversationHandler.END
                return await func(*args, **kwargs)

            else:  # method with self
                self = args[0]
                update = args[1]
                context = args[2]
                user_id = update.effective_user.id
                if user_id not in ALLOWED_USERS:
                    await update.message.reply_text("🚫 You're not authorized to use this bot.")
                    return ConversationHandler.END
                return await func(self, update, context, *args[3:], **kwargs)
        return wrapper
    
