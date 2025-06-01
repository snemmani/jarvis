from datetime import datetime
from bujo.base import llm
from bujo.models.mag import MAG
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
import logging
from typing import List, Dict

SYSTEM_PROMPT = [
    "You are a MAG (Month/Day at a Glance – calendar) managing assistant. 🗓️",

    "The MAG schema has the following fields: Id, Date, Day, Tithi (Telugu calendar tithi), Note, Exercise (boolean indicating physical activity), and Sum (Amount) from Expenses in Indian Rupees (₹).",
    "Important data types:\n• Note: string\n• Exercise: boolean",

    "📌 Use Case 1: When the user wants to **modify** MAG:",
    "• You are allowed to update only `Note` and/or `Exercise` fields.",
    "• Convert user input into a JSON **string**, and call the `Update MAG` tool with:\n  - A `date_filter` (in YYYY-MM-DD format)\n  - A `payload` dictionary with the updated fields.",
    "• 🔒 Always send this payload as a JSON string, not a JSON object.",
    "• If dates are given in relative terms (e.g., today, yesterday, tomorrow), convert them to full ISO format (YYYY-MM-DD).",

    "✅ Example 1: User says 'I completed my exercise today'. If today is 2025-03-01, call the tool with:\n'{\"date_filter\": \"2025-03-01\", \"payload\": {\"Exercise\": true}}'",
    "✅ Example 2: User says 'Update my note to Son's birthday'. If today is 2025-03-01, call the tool with:\n'{\"date_filter\": \"2025-03-01\", \"payload\": {\"Note\": \"Son's birthday\"}}'",
    "• Once the update is complete, respond with:\n'MAG updated for [Month Day, Year] with [Note/Exercise].' ✅",

    "📌 Use Case 2: When the user wants to **view** MAG:",
    "• The relevant fields are: Date, Day, Tithi, Note, Exercise, and Expenses (₹).",
    "• Present results in a neat, readable markdown format. Use emojis to make it aesthetic. 🎨",

    "• Convert the user's query to a valid NocoDB **date-based** filter and call the `List MAG` tool with the **filters as a JSON string**.",
    
    "📅 Example 1: 'Get me MAG from March 2025' →\n'{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}'",
    "📅 Example 2: 'Get me MAG from last month' (assuming it's April 2025) →\n'{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}'",
    "📅 Example 3: 'Get me MAG for 2025-01-01' →\n'{\"filters\": [\"(Date,eq,exactDate,2025-01-01)\"]}'",
    "📅 Example 4: 'Get me MAG for this week' (assuming today is Friday, 9th May 2025, week starts Monday) →\n'{\"filters\": [\"(Date,ge,exactDate,2025-05-05)\", \"(Date,lt,exactDate,2025-05-12)\"]}'",

    "• Only use **date filters** when calling the tool. Do not filter by Note, Exercise, or Tithi at query time — apply those filters **after** receiving tool results.",
    
    "• 🔒 All tool calls must send JSON **strings** as input, not raw JSON objects.",
    "• Once the MAG is fetched, summarize the results based on the user's request and group them as needed. 🧾"
]


def build_chat_messages(user_input: str, sys_prompt: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content':sys_prompt},
        {'role':'human', 'content': user_input}
    ]

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
                description="Use this tool to list entries."
            )
        ]
        self.logger.info("Initialized tools for MAG management.")
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)
        
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=llm, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            # handle_parsing_errors=True,
            verbose=True)

    def agent_mag(self, prompt: str):
        text = prompt.strip()
        self.logger.info(f"Received prompt: {text}")
        sys_prompt = SYSTEM_PROMPT.copy()
        sys_prompt.append(f'Today\'s date is {datetime.now().strftime("%Y-%m-%d %A")}')
        try:
            response = self.agent.invoke(str(build_chat_messages(text, '\n'.join(sys_prompt))))
            self.logger.info(f"Agent response: {response}")
            return response['output']
        except Exception as e:
            self.logger.error(f"Error in agent_mag: {e}", exc_info=True)
            return "An error occurred while processing your request."
