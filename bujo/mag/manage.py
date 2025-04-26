
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler
from datetime import datetime
import json
import os
from bujo.models.expenses import Expenses
from bujo.base import ALLOWED_USERS, COHERE_MODEL, cohere_client
from bujo.models.mag import MAG


# Conversation states
UPDATE_MAG, LIST_MAG_CHAT = range(2)

class MagManager:
    def __init__(self, mag_model: MAG):
        """
        Create an Object of ExpenseManager class.

        :param  expenses_model: NocoDB Expenses Model Instance
        """
        self.mag_model = mag_model
        self.messages_modify = [
            {'role': 'system', 'content': '''## LLM ROLE: You are an expert freetext to python serializer. These are fields in MAG (Month/Day at a Glance) schema that you're going to update. Either Note or Exercise ##\n## INSTRUCTION: Note: String type, Exercise: Boolean type ##\n## INSTRUCTION: Convert whatever data that I provide into a JSON. There should only be the JSON, nothing else in the response ##\n## INSTRUCTION: If the dates are provided in relative terms i.e yesterday, tomorrow etc.. convert them to  full ISO format ##\n## INSTRUCTION: If user provides a freestyle text, try to convert it into above schema.\n## INSTRUCTION: For example if user says something like 'I completed my exercise today. Assuming today's date is 2025-03-01. You respond with {"date_filter": "2025-03-01", "payload": {"Exercise": true}}\n## INSTRUCTION: For example if user says something like "Update my note to Sony's birthday". Assuming today's date is 2025-03-01. You respond with {"date_filter": "2025-03-01", "payload": {"Note": "Sony's birthday"}}\n## INSTRUCTION: For example if user says something like "Update my note to Sony's birthday and mark my exercise as done today". Assuming today's date is 2025-03-01. You respond with {"date_filter": "2025-03-01", "payload": {"Note": "Sony's birthday", "Exercise": true}}\n## Coming to present context'''}
        ]

        self.messages_list = [
            {'role': 'system', 'content': """## LLM ROLE: You are an expert freetext to python serializer and my personal interpretor. There are following fields in MAG schema: Id, Date, Day, Tithi (translates to telugu calender tithi), Note, Exercise (Indicates physical activity done or not), and Sum(Amount) from Expenses in Indian Rupees. ##\n## INSTRUCTION: You have 1 roles as a part of this exercise  ##\n## INSTRUCTION: When I say ACT_AS_DATE_HELPER, you will take the input from me and then convert my query to a nocodb filter condition   ##\n## ACT_AS_DATE_HELPER Instruction: For example, if I asked Get me MAG from the month of march 2025, you will respond with ["(Date,ge,exactDate,2025-03-1)", "(Date,lt,exactDate,2025-04-01)"] ##\n## ACT_AS_DATE_HELPER Instruction: Another example, if I asked Get me MAG from last month, you would check which month last month is, then give me a condition like ["(Date,ge,exactDate,2025-03-01)", "(Date,lt,exactDate,2025-04-01)"]" ##\n## ACT_AS_DATE_HELPER Instruction: Another example, if I asked Get me MAG for a specific day ex. 2025-01-01, then you will provide the filter as ["(Date,eq,exactDate,2025-01-01)"] ##\n## ACT_AS_DATE_HELPER Instruction: You will respond with the Filter query only and nothing else ##\n"""}
        ]

    async def start_modify(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        await update.message.reply_text("💰 Update your Mag by giving instructions to modify a note or exercise done below. Done with changes? Go for cancel command!")
        return UPDATE_MAG
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚫 MAG interaction cancelled.")
        return ConversationHandler.END

    async def handle_mag_change(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        text = update.message.text.strip()

        messages_copy = self.messages_modify.copy()
        today = datetime.now().strftime("%Y-%m-%d %A")  # e.g. "April 11, 2025"
        messages_copy.append({'role': 'system', 'content': f"Today's date is {today}"})
        messages_copy.append({'role': 'user', 'content': text})

        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages = messages_copy
        )

        payload = json.loads(gen_ai_response.message.content[0].text.replace('```json', '').replace('```', ''))
        if not payload or 'date_filter' not in payload:
            await update.message.reply_text("❌ Failed to parse the input. Try again?")
            return UPDATE_MAG        

        mag_instance = self.mag_model.find_by_date(payload['date_filter'])
        if not mag_instance:
            await update.message.reply_text("❌ No MAG object found for the given date. Try again?")
            return UPDATE_MAG
        
        for key, value in payload['payload'].items():
            mag_instance[key] = value
        
        
        self.mag_model.update(mag_instance)
        allowed_keys = {"Id", "Date", "Note", "Tithi", "Exercise", "Sum(Amount) from Expenses"}
        filtered_data = {key: value for key, value in mag_instance.items() if key in allowed_keys}
        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages = [{"role": "user", "content": "Summarise the following data: " + str(filtered_data) + "\n## INSTRUCTION: Currency is always in Indian Rupees with Symbol '₹'"}]
        )

        response_text = gen_ai_response.message.content[0].text
        
        await update.message.reply_text(
            f"✅ MAG updated successfully! Here's the summary:\n\n{response_text}",
            parse_mode='markdown'
        )
        
        return UPDATE_MAG


    async def start_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        await update.message.reply_text(
            "📅 Please specify the month (e.g., 'March 2023') or day (e.g., '2023-03-15') for which you want to list MAG entries."
        )
        return LIST_MAG_CHAT

    async def handle_list_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("🚫 You're not authorized to use this bot.")
            return ConversationHandler.END

        text = update.message.text.strip()

        messages_copy = self.messages_list.copy()
        messages_copy.append({'role': 'system', 'content': f"Today's date is {datetime.now().strftime('%Y-%m-%d %A')}"})  # e.g. "April 11, 2025"
        messages_copy.append({'role': 'user', 'content': text})

        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages = messages_copy
        )

        # Parse the input to determine if it's a month or a specific day
        try:
            payload = json.loads(gen_ai_response.message.content[0].text.replace('```json', '').replace('```', ''))
            mag_entries = self.mag_model.list(where=payload)

            if not mag_entries:
                await update.message.reply_text(f"❌ No entries found for the specified period. {payload}")
                return LIST_MAG_CHAT
                
        except Exception:
            await update.message.reply_text(f"❌ Failed to fetch MAG entries for payload {payload}.")
            return LIST_MAG_CHAT
        

        # Summarize the expenses using Cohere
        expense_texts = [
            f"Date: {mag['Date']}, Day: {mag['Day']}, Amount: {mag['Tithi']}, Note: {mag['Note']}, Exercise: {mag['Exercise']}, Expenses on day: {mag['Sum(Amount) from Expenses']}"
            for mag in mag_entries
        ]
        summary_prompt = (
            f"Summarize the following MAG (Month at a glance) data by Date for the specified period.\n Provide the results as html output with complete html, like doctype and everything to support all encodings and languages, you should also provide styling to give aesthetic display to tables.\n Provide a summary of total expenses in the last line.\n For the exercise done or not done, indicate a checkmark type emoji instead of True or False.\n All of this without the ```html banner. \n {payload}.\n ## INSTRUCTION: Currency is always in Indian Rupees with Symbol '₹'\n" +
            "\n".join(expense_texts)
        )
        gen_ai_response = cohere_client.chat(
            model=COHERE_MODEL, 
            messages=[{'role': 'user', 'content': summary_prompt}]
        )

        summary = gen_ai_response.message.content[0].text.strip()
        # Save the summary to a temporary file and send it as a document
        doc_name = os.path.join(os.environ['TEMP'], 'mag_summary.html')
        with open(doc_name, 'w', encoding='utf-8-sig') as f:
            f.write(summary)
        await update.message.reply_document(doc_name, caption="MAG Summary")
        # await update.message.reply_text(
        #     f"📋 MAG:\n\n{summary}",
        #     parse_mode='markdown'
        # )
        return LIST_MAG_CHAT
    