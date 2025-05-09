
from bujo.models.expenses import Expenses
from bujo.models.mag import MAG
from bujo.base import llm
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import json


SYSTEM_PROMPT = [
    "You are an expense tracking assistant. "
    "Case 1: When a user wants to add an expense, follow below instructions:"
    "Extract 'Item', 'Amount', and 'Date' (YYYY-MM-DD format) from user inputs like 'Add mangoes for 40 rupees today'. "
    "Call the tool `add_expense` with a dictionary {Item, Amount, Date}.",
    "Once the expense is added, respond with 'Expense added on Date for Item ₹Amount'.",
    "Case 2: When a user want to list expenses, follow below instructions:",
    """If I ask "Get me expenses from the month of March 2025", you will call the expense lookup tool with {"filters": ["(Date,ge,exactDate,2025-03-01)", "(Date,lt,exactDate,2025-04-01)"]}.""",
    """If I ask "Get me expenses from last month", compute the correct and for example assume month is may 2025 and you will call the expense lookup tool with {"filters": ["(Date,ge,exactDate,2025-05-01)", "(Date,lt,exactDate,2025-06-01)"]}.""",
    """If I ask "Get me expenses for this week", compute the correct week [Week starts with Monday] and for example assume today is 9th May 2025, Friday, then and you will call the expense lookup tool with {"filters": ["(Date,ge,exactDate,2025-05-05)", "(Date,lt,exactDate,2025-05-12)"]}.""",
    """If I ask "Get me expenses for 2025-01-01", you will call the expense lookup tool with {"filters": ["(Date,eq,exactDate,2025-01-01)"]}.""",
    """The tool input should only be the date filters, do not consider Item or Amount for filters as input to tool, but performing filtering from them after the tool has responded."""
    "Once the expenses are fetched by the tool, summarize the expenses based on the users request and grouping requirements"
]

def prepend_system_prompt(user_input: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {user_input}"

class ExpenseManager:
    def __init__(self, expenses_model: Expenses, mag_model: MAG):
        """
        Create an Object of ExpenseManager class.

        :param  expenses_model: NocoDB Expenses Model Instance
        """
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

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=llm, 
            agent="chat-conversational-react-description", 
            memory=self.memory,
            # handle_parsing_errors=True,
            verbose=True)

    def agent_expenses(self, prompt: str):
        text = prompt.strip()
        response = self.agent.run(prepend_system_prompt(text))
        return response
    
    def add_expense(self, data: str):
        """
        Add an expense to the database.
        
        :param data: JSON string containing the expense data.
        :return: Response from the expenses model.
        """
        data_object = json.loads(data)
        response_add = self.expenses_model.create(data_object)
        if 'failed' in str(response_add).lower():
            return "Failed to add expense entry. Try again?"
        mag_object = self.mag_model.find_by_date(data_object['Date'])
        if mag_object:
            self.expenses_model.link_mag_to_expense(response_add.json().get("Id"), mag_object['Id'])
        return response_add
    