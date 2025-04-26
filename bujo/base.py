import cohere
import os
from bujo.models.expenses import Expenses
from bujo.models.mag import MAG

# Globals
ALLOWED_USERS = [int(os.environ["TELEGRAM_USER_ID"])]
COHERE_MODEL = "command-a-03-2025"
NOCODB_BASE_URL = "https://app.nocodb.com"

# Secrets
COHERE_API_KEY = os.environ['COHERE_API_KEY']
NOCODB_API_TOKEN = os.environ['NOCODB_API_TOKEN']
NOCODB_EXPENSES_TABLE_ID = os.environ["NOCODB_EXPENSES_TABLE_ID"]
NOCODB_MAG_TABLE_ID = os.environ["NOCODB_MAG_TABLE_ID"]
NOCODB_EXPENSES_MAG_LINK_ID = os.environ["NOCODB_EXPENSES_MAG_LINK_ID"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

# Initializations
cohere_client = cohere.ClientV2(COHERE_API_KEY)
mag_model = MAG(NOCODB_BASE_URL, NOCODB_API_TOKEN, NOCODB_MAG_TABLE_ID)
expenses_model = Expenses(NOCODB_BASE_URL, NOCODB_API_TOKEN, NOCODB_EXPENSES_TABLE_ID, NOCODB_EXPENSES_MAG_LINK_ID, mag_model)
expense_add_messages=[
    {'role': 'system', 'content': 'You are an expert freetext to python serializer'},
    {'role': 'system', 'content': 'There are following fields in expenses schema: date, item, amount'},
    {'role': 'system', 'content': 'Your role is to convert whatever data that I provide into a JSON. You will respond witha  json object/list only. Nothing else.'},
]
    