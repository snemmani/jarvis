
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler
from datetime import datetime
import json
from bujo.models.expenses import Expenses
from bujo.base import ALLOWED_USERS, COHERE_MODEL, cohere_client


# Conversation states
ADD_EXPENSE_CHAT, LIST_EXPENSE_CHAT = range(2)

class ExpenseManager:
    def __init__(self, expenses_model: Expenses):
        """
        Create an Object of ExpenseManager class.

        :param  expenses_model: NocoDB Expenses Model Instance
        """
        self.expenses_model = expenses_model
        today = datetime.now().strftime("%Y-%m-%d %A")  # e.g. "April 11, 2025"
        self.messages_add = [
            {'role': 'system', 'content': f'## LLM ROLE: You are an expert freetext to python serializer. There are following fields in expenses schema: Date, Item, Amount. ##\n## INSTRUCTION: Convert whatever data that I provide into a JSON. There should only be the JSON, nothing else in the response ##\n## INSTRUCTION: If the dates are provided in relative terms i.e yesterday, tomorrow etc.. convert them to  full ISO format ##\n## INSTRUCTION: The dates will and should always be from the past, if a future date is given, substitute today\'s date ##\n## If user provides a freestyle text, try to convert it into above schema.\nToday is {today} ##'}
        ]

        self.messages_list = [
            {'role': 'system', 'content': f"""## LLM ROLE: You are an expert freetext to python serializer and my personal interpretor. There are following fields in expenses schema: Date, Item, Amount. ##\n## INSTRUCTION: You have 1 roles as a part of this exercise  ##\n## INSTRUCTION: When I say ACT_AS_DATE_HELPER, you will take the input from me and then convert my query to a nocodb filter condition   ##\n## INSTRUCTION: Remember that today's date is {today}\n## ACT_AS_DATE_HELPER Instruction: For example, if I asked Get me expenses from the month of march 2025, you will respond with ["(Date,ge,exactDate,2025-03-1)", "(Date,lt,exactDate,2025-04-01)"] ##\n## ACT_AS_DATE_HELPER Instruction: Another example, if I asked Get me expenses from last month, you would check which month last month is, then give me a condition like ["(Date,ge,exactDate,2025-03-01)", "(Date,lt,exactDate,2025-04-01)"]" ##\n## ACT_AS_DATE_HELPER Instruction: Another example, if I asked Get me expenses for a specific day ex. 2025-01-01, then you will provide the filter as ["(Date,eq,exactDate,2025-01-01)"] ##\n## ACT_AS_DATE_HELPER Instruction: You will respond with the Filter query only and nothing else ##\n## INSTRUCTION: If the user asks about a date from future, in any way, keep the end date as today or exact date as today. ##\n"""}
        ]

    async def start_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        await update.message.reply_text("💰 Add your expenses below. Done adding them? Go for cancel command!")
        return ADD_EXPENSE_CHAT
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚫 Expenses interaction cancelled.")
        return ConversationHandler.END

    async def handle_expense_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        text = update.message.text.strip()

        messages_copy = self.messages_add.copy()
        messages_copy.append({'role': 'user', 'content': text})

        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages = messages_copy
        )

        payload = json.loads(gen_ai_response.message.content[0].text.replace('```json', '').replace('```', ''))

        mag_instance = self.expenses_model.mag_table_instance.find_by_date(payload['Date'])
        if not mag_instance:
            await update.message.reply_text("❌ No MAG object found for the given date. Try again?")
            return ADD_EXPENSE_CHAT
        
        expense_instance = self.expenses_model.create(payload)
        if not expense_instance:
            await update.message.reply_text("❌ Failed to create expense entry. Try again?")
            return ADD_EXPENSE_CHAT
        
        final_response = self.expenses_model.link_mag_to_expense(expense_instance['Id'], mag_instance['Id'])   
        if not final_response:
            await update.message.reply_text("❌ Failed to link expense to MAG. Try again?")
            return ADD_EXPENSE_CHAT
        await update.message.reply_text(
            f"✅ Expense entry added and linked successfully!\n\n"
            f"📅 Date: {payload['Date']}\n"
            f"🛒 Item: {payload['Item']}\n"
            f"💵 Amount: {payload['Amount']}"
        )
        return ADD_EXPENSE_CHAT


    async def start_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        await update.message.reply_text(
            "📅 Please specify the month (e.g., 'March 2023') or day (e.g., '2023-03-15') for which you want to list expenses."
        )
        return LIST_EXPENSE_CHAT

    async def handle_list_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        text = update.message.text.strip()

        messages_copy = self.messages_list.copy()
        messages_copy.append({'role': 'user', 'content': text})

        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages = messages_copy
        )

        # Parse the input to determine if it's a month or a specific day
        try:
            payload = json.loads(gen_ai_response.message.content[0].text.replace('```json', '').replace('```', ''))
            expenses = self.expenses_model.list(where=payload)

            if not expenses:
                await update.message.reply_text(f"❌ No expenses found for the specified period. {payload}")
                return LIST_EXPENSE_CHAT
                
        except Exception:
            await update.message.reply_text(f"❌ Failed to fetch expenses for payload {payload}.")
            return LIST_EXPENSE_CHAT
        

        # Summarize the expenses using Cohere
        expense_texts = [
            f"Date: {expense['Date']}, Item: {expense['Item']}, Amount: {expense['Amount']}"
            for expense in expenses
        ]
        summary_prompt = (
            f"Summarize the following expenses data by Date and Category for the specified period: {payload}.\n ## INSTRUCTION: Currency is always in Indian Rupees with Symbol '₹'\n" +
            "\n".join(expense_texts)
        )
        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages=[{'role': 'user', 'content': summary_prompt}]
        )

        summary = gen_ai_response.message.content[0].text.strip()

        await update.message.reply_text(
            f"📋 Expenses Summary:\n\n{summary}",
            parse_mode='markdown'
        )
        return LIST_EXPENSE_CHAT







# NocoDB config (replace with actual values or import from settings file)




# Allowed user IDs








    