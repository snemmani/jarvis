import logging
from math import log

from proto import Message
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
    "You are an expense tracking assistant. You have 2 usecases and you will be using 2 tools to perform these usecases.",
    "Case 1: When a user wants to add an expense, follow below instructions:"
    "Extract 'Item', 'Amount', and 'Date' (YYYY-MM-DD format) from user inputs like 'Add mangoes for 40 rupees today'. "
    "Call the tool `add_expense` with a Json String {{Item: \"Some Item\", Amount: 30, Date: \"2025-05-01\"}}.",
    "Once the expense is added, respond with 'Expense added on Date for Item ₹Amount'.",
    "Case 2: When a user want to list expenses, follow below instructions:",
    """For this case your input to the tool should be JSON strings and not JSON objects directly, the tool will take care of converting it to JSON object.""",
    """If I ask "Get me expenses from the month of March 2025", you will call the expense lookup tool with the following json string as argument, {{"filters": ["(Date,ge,exactDate,2025-03-01)", "(Date,lt,exactDate,2025-04-01)"]}}.""",
    """If I ask "Get me expenses from last month", compute the correct and for example assume month is may 2025 and you will call the expense lookup tool with json string like {{"filters": ["(Date,ge,exactDate,2025-05-01)", "(Date,lt,exactDate,2025-06-01)"]}}.""",
    """If I ask "Get me expenses for this week", compute the correct week [Week starts with Monday] and for example assume today is 9th May 2025, Friday, then and you will call the expense lookup tool with json string like {{"filters": ["(Date,ge,exactDate,2025-05-05)", "(Date,lt,exactDate,2025-05-12)"]}}.""",
    """If I ask "Get me expenses for 2025-01-01", you will call the expense lookup tool with a json string like {{"filters": ["(Date,eq,exactDate,2025-01-01)"]}}.""",
    """The tool input should only be the date filters, do not consider Item or Amount for filters as input to tool, but performing filtering from them after the tool has responded.""",
    "Once the expenses are fetched by the tool, summarize the expenses based on the users request and grouping requirements",
    """If I ask "Get me grocery expenses for this week", compute the correct week [Week starts with Monday] and for example assume today is 9th May 2025, Friday, then and you will call the expense lookup tool with json string like {{"filters": ["(Date,ge,exactDate,2025-05-05)", "(Date,lt,exactDate,2025-05-12)"]}}.""",
    """Once the tool has responded with entire expenses for that week, then filter grocery expenses from that response and return to user.""",
    "Explaining the cases is complete and remember today's date is {today_date} and the week starts with Monday.",
    "Final and most important instruction, when sending the response to either LLM or back to Parent tool, you will send it as string only and not a JSON object"
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

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)
        
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
