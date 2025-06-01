import logging
from math import log

from bujo.models.expenses import Expenses
from bujo.models.mag import MAG
from bujo.base import llm
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_types import AgentType
from datetime import datetime
import json
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expense tracking assistant. You handle two use cases and use two tools to perform them.",

    "Use Case 1: When a user wants to **add an expense**, follow these instructions:",
    "• Extract 'Item', 'Amount', and 'Date' (in YYYY-MM-DD format) from user inputs like 'Add mangoes for 40 rupees today'.",
    "• Call the tool `add_expense` with a JSON string in the format: {{\"Item\": \"Some Item\", \"Amount\": 30, \"Date\": \"2025-05-01\"}}.",
    "• Once the expense is added, respond with: '✅ Expense added on Date for Item ₹Amount'.",

    "Use Case 2: When a user wants to **list expenses**, follow these instructions:",
    "• Always send tool inputs as **JSON strings**, not raw JSON objects. The tool will handle the conversion.",
    "• Example: If I ask 'Get me expenses from the month of March 2025', call the lookup tool with a json string like this as argument: {{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}}.",
    "• If I ask 'Get me expenses from last month' and today's date is May 2025, use this json string: {{\"filters\": [\"(Date,ge,exactDate,2025-04-01)\", \"(Date,lt,exactDate,2025-05-01)\"]}}.",
    "• If I ask 'Get me expenses for this week', and today is Friday, 9th May 2025, use this json string: {{\"filters\": [\"(Date,ge,exactDate,2025-05-05)\", \"(Date,lt,exactDate,2025-05-12)\"]}}.",
    "• If I ask 'Get me expenses for 2025-01-01', use this json string: {{\"filters\": [\"(Date,eq,exactDate,2025-01-01)\"]}}.",
    "• Only include date filters in the tool input. Do not filter by Item or Amount in the input — apply those filters **after** you receive the tool response.",
    "• Once expenses are fetched, summarize them according to the user's request and any grouping mentioned.",
    "• Example: If I ask 'Get me grocery expenses for this week' and today is 9th May 2025 (Friday), call the tool with: {{\"filters\": [\"(Date,ge,exactDate,2025-05-05)\", \"(Date,lt,exactDate,2025-05-12)\"]}}. Then filter out grocery expenses from the result and present them.",

    "Remember that today's date is {today_date}, and the week starts on Monday.",

    "📌 Final and most important instruction: When sending responses to either the LLM or the parent tool, always send them as a **string** — not a JSON object.",
    
    "🧾 Format all tool outputs cleanly using **markdown**. Separate multiple expenses or results using new lines.",
    "✨ Use emojis where appropriate to enhance readability and make the response friendly."
]


class ExpenseManager:
    def __init__(self, expenses_model: Expenses, mag_model: MAG):
        """
        Create an Object of ExpenseManager class.

        :param  expenses_model: NocoDB Expenses Model Instance
        """
        logger.info("Initializing ExpenseManager")
        self.expenses_model = expenses_model
        self.mag_model = mag_model
        self.tools = [
            Tool(
                name="Expense Lookup",
                func=self.expenses_model.list,
                description="Use this tool to fetch expenses based on specific filters."
            ),
            Tool(
                name="Expense Creation",
                func=self.add_expense,
                description="Use this tool to create a new expense entry."
            )
        ]

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=1)
        
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=llm, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            memory=self.memory,
            # handle_parsing_errors=True,
            verbose=True)
        
        self.prompt = ChatPromptTemplate.from_messages([
            { "role": "system", "content": '\n'.join(SYSTEM_PROMPT)},
            { 'role' : 'human', 'content': "{input}" } 
        ])

        self.chain = self.prompt | self.agent
        logger.info("ExpenseManager initialized successfully")

    def agent_expenses(self, prompt: str):
        text = prompt.strip()
        logger.info(f"Received prompt: {text}")
        try:
            response = self.agent.invoke(str(self.prompt.format_messages(input=text, today_date=datetime.now().strftime("%Y-%m-%d %A"))))
            logger.info(f"Agent response: {response}")
            return response['output']
        except Exception as e:
            logger.error(f"Error in agent_expenses: {e}", exc_info=True)
            return "Sorry, there was an error processing your request."

    def add_expense(self, data: str):
        """
        Add an expense to the database.
        
        :param data: JSON string containing the expense data.
        :return: Response from the expenses model.
        """
        logger.info(f"Adding expense with data: {data}")
        try:
            data_object = json.loads(data)
            response_add = self.expenses_model.create(data_object)
            logger.info(f"Expense creation response: {response_add}")
            if 'failed' in str(response_add).lower():
                logger.warning("Failed to add expense entry.")
                return "Failed to add expense entry. Try again?"
            mag_object = self.mag_model.find_by_date(data_object['Date'])
            if mag_object:
                logger.info(f"Linking MAG {mag_object['Id']} to expense {response_add['Id']}")
                self.expenses_model.link_mag_to_expense(response_add["Id"], mag_object['Id'])
            else:
                logger.info("No MAG entry found to link with the expense.")
                return "Expense added but no MAG entry found to link the expense to that date!"
            return f"Expense added successfully"
        except Exception as e:
            logger.error(f"Error in add_expense: {e}", exc_info=True)
            return "Failed to add expense entry due to an error."
