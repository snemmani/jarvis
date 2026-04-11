import logging
from datetime import datetime

from bujo.models.mag import MAG
from bujo.base import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are a MAG (Month/Day at a Glance – calendar) managing assistant. 🗓️",
    "⚠️ CRITICAL: You MUST always call a tool to complete any request. NEVER respond with a fabricated or assumed result. If the user wants to update MAG, you MUST call `Update_MAG`. If the user wants to view MAG, you MUST call `List_MAG`. Do not skip tool calls under any circumstances.",

    "The MAG schema has the following fields: Id, Date, Day, Tithi (Telugu calendar tithi), Note, Exercise (boolean indicating physical activity), and Sum (Amount) from Expenses in Indian Rupees (₹).",
    "Important data types:\n• Note: string\n• Exercise: boolean",

    "📌 Use Case 1: When the user wants to **modify** MAG:",
    "• You are allowed to update only `Note` and/or `Exercise` fields.",
    "• Convert user input into a JSON **string**, and call the `Update_MAG` tool with:\n  - A `date_filter` (in YYYY-MM-DD format)\n  - A `payload` dictionary with the updated fields.",
    "• 🔒 Always send this payload as a JSON string, not a JSON object.",
    "• If dates are given in relative terms (e.g., today, yesterday, tomorrow), convert them to full ISO format (YYYY-MM-DD).",

    "✅ Example 1: User says 'I completed my exercise today'. If today is 2025-03-01, call the tool with:\n'{\"date_filter\": \"2025-03-01\", \"payload\": {\"Exercise\": true}}'",
    "✅ Example 2: User says 'Update my note to Son's birthday'. If today is 2025-03-01, call the tool with:\n'{\"date_filter\": \"2025-03-01\", \"payload\": {\"Note\": \"Son's birthday\"}}'",
    "• Once the update is complete, respond with:\n'MAG updated for [Month Day, Year] with [Note/Exercise].' ✅",

    "📌 Use Case 2: When the user wants to **view** MAG:",
    "• The relevant fields are: Date, Day, Tithi, Note, Exercise, and Expenses (₹).",
    "• Present results in a neat, readable markdown format. Use emojis to make it aesthetic. 🎨",
    "• Convert the user's query to a valid NocoDB **date-based** filter and call the `List_MAG` tool with the **filters as a JSON string**.",

    "📅 Example 1: 'Get me MAG from March 2025' →\n'{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}'",
    "📅 Example 2: 'Get me MAG from last month' (assuming it's April 2025) →\n'{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}'",
    "📅 Example 3: 'Get me MAG for 2025-01-01' →\n'{\"filters\": [\"(Date,eq,exactDate,2025-01-01)\"]}'",
    "📅 Example 4: 'Get me MAG for this week' (assuming today is Friday, 9th May 2025, week starts Monday) →\n'{\"filters\": [\"(Date,ge,exactDate,2025-05-05)\", \"(Date,lt,exactDate,2025-05-12)\"]}'",

    "• Only use **date filters** when calling the tool. Do not filter by Note, Exercise, or Tithi at query time — apply those filters **after** receiving tool results.",
    "• 🔒 All tool calls must send JSON **strings** as input, not raw JSON objects.",
    "• Once the MAG is fetched, summarize the results based on the user's request and group them as needed. 🧾",
    "💰 All monetary amounts are in Indian Rupees. Always display amounts with the ₹ symbol — never use $ or any other currency symbol.",
]


class MagManager:
    def __init__(self, mag_model: MAG):
        self.mag_model = mag_model

        tools = [
            Tool(
                name="Update_MAG",
                func=self.mag_model.update,
                description="Use this tool to update MAG.",
            ),
            Tool(
                name="List_MAG",
                func=self.mag_model.list,
                description="Use this tool to list MAG entries.",
            ),
        ]

        _memory = MemorySaver()

        def _state_modifier(state):
            today = datetime.now().strftime("%Y-%m-%d %A")
            content = "\n".join(SYSTEM_PROMPT) + f"\nToday's date is {today}."
            return [SystemMessage(content=content)] + state["messages"]

        self.agent = create_react_agent(llm, tools, prompt=_state_modifier, checkpointer=_memory)
        logger.info("MagManager initialized.")

    def agent_mag(self, prompt: str) -> str:
        text = prompt.strip()
        logger.info("Received prompt: %s", text)
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": "mag"}},
            )
            output = result["messages"][-1].content
            logger.info("Agent response: %s", output)
            return output
        except Exception as e:
            logger.error("Error in agent_mag: %s", e, exc_info=True)
            return "An error occurred while processing your request."
