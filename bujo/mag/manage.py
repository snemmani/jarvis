from datetime import datetime
from bujo.base import llm
from bujo.models.mag import MAG
from langchain.agents import initialize_agent, Tool
from datetime import datetime
from langchain.memory import ConversationBufferMemory
import logging

SYSTEM_PROMPT = [
    "You are an MAG (Calendar) managing assistant. "
    '''There are following fields in MAG schema: Id, Date, Day, Tithi (translates to telugu calender tithi), Note, Exercise (Indicates physical activity done or not), and Sum(Amount) from Expenses in Indian Rupees.''',
    'Data Types of the fields that are important to you are -> Note: String type, Exercise: Boolean type',
    "Case 1: When a user wants to modify MAG follow below instructions:",
    '''These are fields in MAG (Month/Day at a Glance) schema that you're going to update. Either Note or Exercise''',
    "Convert whatever data that I provide into a JSON and call the Update MAG tool with the JSON string.",
    'If the dates are provided in relative terms i.e yesterday, tomorrow etc.. convert them to  full ISO format like YYYY-MM-DD', 
    '''For example if user says something like 'I completed my exercise today. Assuming today's date is 2025-03-01. You call the Update MAG tool with {"date_filter": "2025-03-01", "payload": {"Exercise": true}}''',
    '''For example if user says something like "Update my note to Son's birthday". Assuming today's date is 2025-03-01. You respond with {"date_filter": "2025-03-01", "payload": {"Note": "Sony's birthday"}}''',
    'Once the update is complete, respond with "MAG updated for date Month, Day Year with Note/Exercise',
    "Case 2: When a user wants to view MAG, follow below instructions:",
    "Interested fields the user wants to view are Date, Day, Tithi, Note, Exercise and Expenses (In Rupees)",
    "Format the response in a neat format using emoticons wherever possible",
    """You will take the input from me and then convert my query to a nocodb filter condition and call the List MAG tool""",
    'For example, if I asked Get me MAG from the month of march 2025, you will call the tool with {"filters": ["(Date,ge,exactDate,2025-03-1)", "(Date,lt,exactDate,2025-04-01)"]}',
    'Another example, if I asked Get me MAG from last month, you would check which month last month is, then call the tool with  { "fiters": ["(Date,ge,exactDate,2025-03-01)", "(Date,lt,exactDate,2025-04-01)"]"} assuming this month is April 2025',
    'Another example, if I asked Get me MAG for a specific day ex. 2025-01-01, then you will call tool with {"filters": ["(Date,eq,exactDate,2025-01-01)"]}', 
    '''Another example, if I asked Get me MAG for this week, you will check which week it is, and then call the tool with {"filters": ["(Date,ge,exactDate,2025-05-05)", "(Date,lt,exactDate,2025-05-12)"]} assuming today is 9th May 2025 which is Friday and week starts with Monday''',
    'You will call the tool with date filters only, do not consider Exercise or Note or Tithi for filters as input to tool, but perform filtering from them after the tool has responded.',
    'Once the MAG is fetched by the tool, summarize the MAG based on the users request and grouping requirements'
]

def prepend_system_prompt(user_input: str) -> str:
    date_today = datetime.now().strftime("%Y-%m-%d %A")  # e.g. "April 11, 2025"
    return f"{SYSTEM_PROMPT}\nToday's date is {date_today}\n\nUser: {user_input}"

class MagManager:
    def __init__(self, mag_model: MAG):
        """
        Create an Object of ExpenseManager class.

        :param  expenses_model: NocoDB Expenses Model Instance
        """
        self.mag_model = mag_model

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.tools = [
            Tool(
                name="Update MAG",
                func=self.mag_model.update,
                description="Use this tool to update MAG."
            ),
            Tool(
                name="List MAG",
                func=self.mag_model.list,
                description="Use this tool to create a new expense entry."
            )
        ]
        self.logger.info("Initialized tools for MAG management.")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=llm, 
            agent="chat-conversational-react-description", 
            memory=self.memory,
            # handle_parsing_errors=True,
            verbose=True)

    def agent_mag(self, prompt: str):
        text = prompt.strip()
        self.logger.info(f"Received prompt: {text}")
        try:
            response = self.agent.run(prepend_system_prompt(text))
            self.logger.info(f"Agent response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error in agent_mag: {e}", exc_info=True)
            return "An error occurred while processing your request."
