import json
import logging
from datetime import datetime

from bujo.models.expenses import Expenses
from bujo.models.mag import MAG
from bujo.base import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expense tracking assistant. You handle two use cases and use two tools to perform them.",
    "⚠️ CRITICAL: You MUST always call a tool to complete any request. NEVER respond with a fabricated or assumed result. If the user asks to add an expense, you MUST call `Expense_Creation`. If the user asks to list expenses, you MUST call `Expense_Lookup`. Do not skip tool calls under any circumstances.",

    "Use Case 1: When a user wants to **add an expense**, follow these instructions:",
    "• Extract 'Item', 'Amount', and 'Date' (in YYYY-MM-DD format) from user inputs like 'Add mangoes for 40 rupees today'.",
    "• Call the tool `Expense_Creation` with a JSON string in the format: {{\"Item\": \"Some Item\", \"Amount\": 30, \"Date\": \"2025-05-01\"}}.",
    "• Once the expense is added, respond with: '✅ Expense added on Date for Item ₹Amount'.",

    "Use Case 2: When a user wants to **list expenses**, follow these instructions:",
    "• Always send tool inputs as **JSON strings**, not raw JSON objects. The tool will handle the conversion.",
    "• Example: If I ask 'Get me expenses from the month of March 2025', call the lookup tool with a json string like this as argument: {{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}}.",
    "• If I ask 'Get me expenses from last month' and today's date is May 2025, use this json string: {{\"filters\": [\"(Date,ge,exactDate,2025-04-01)\", \"(Date,lt,exactDate,2025-05-01)\"]}}.",
    "• If I ask 'Get me expenses for this week', and today is Friday, 9th May 2025, use this json string: {{\"filters\": [\"(Date,ge,exactDate,2025-05-05)\", \"(Date,lt,exactDate,2025-05-12)\"]}}.",
    "• If I ask 'Get me expenses for 2025-01-01', use this json string: {{\"filters\": [\"(Date,eq,exactDate,2025-01-01)\"]}}.",
    "• Only include date filters in the tool input. Do not filter by Item or Amount in the input — apply those filters **after** you receive the tool response.",
    "• Once expenses are fetched, summarize them according to the user's request and any grouping mentioned.",

    "Remember that today's date is {today_date}, and the week starts on Monday.",

    "📌 Final and most important instruction: When sending responses to either the LLM or the parent tool, always send them as a **string** — not a JSON object.",
    "💰 All amounts are in Indian Rupees. Always display amounts with the ₹ symbol — never use $ or any other currency symbol.",
    "🧾 Format all tool outputs cleanly using **markdown**. Separate multiple expenses or results using new lines.",
    "✨ Use emojis where appropriate to enhance readability and make the response friendly.",
]


class ExpenseManager:
    def __init__(self, expenses_model: Expenses, mag_model: MAG):
        logger.info("Initializing ExpenseManager")
        self.expenses_model = expenses_model
        self.mag_model = mag_model

        tools = [
            Tool(
                name="Expense_Lookup",
                func=self.expenses_model.list,
                description="Use this tool to fetch expenses based on specific filters.",
            ),
            Tool(
                name="Expense_Creation",
                func=self.add_expense,
                description="Use this tool to create a new expense entry.",
            ),
        ]

        self._tools = tools
        self._build_agent()
        logger.info("ExpenseManager initialized successfully")

    def _build_agent(self):
        def _state_modifier(state):
            today = datetime.now().strftime("%Y-%m-%d %A")
            content = "\n".join(SYSTEM_PROMPT).replace("{today_date}", today)
            return [SystemMessage(content=content)] + state["messages"]

        self.agent = create_react_agent(llm, self._tools, prompt=_state_modifier, checkpointer=MemorySaver())

    def agent_expenses(self, prompt: str) -> str:
        text = prompt.strip()
        logger.info("Received prompt: %s", text)
        for attempt in range(2):
            try:
                result = self.agent.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": "expenses"}},
                )
                output = result["messages"][-1].content
                logger.info("Agent response: %s", output)
                return output
            except ValueError as e:
                if "INVALID_CHAT_HISTORY" in str(e) and attempt == 0:
                    logger.warning("Corrupt expenses chat history — resetting memory and retrying.")
                    self._build_agent()
                    continue
                logger.error("Error in agent_expenses: %s", e, exc_info=True)
                return "Sorry, there was an error processing your request."
            except Exception as e:
                logger.error("Error in agent_expenses: %s", e, exc_info=True)
                return "Sorry, there was an error processing your request."

    def add_expense(self, data: str) -> str:
        logger.info("Adding expense with data: %s", data)
        try:
            data_object = json.loads(data)
            response_add = self.expenses_model.create(data_object)
            logger.info("Expense creation response: %s", response_add)
            if "failed" in str(response_add).lower():
                logger.warning("Failed to add expense entry.")
                return "Failed to add expense entry. Try again?"
            mag_object = self.mag_model.find_by_date(data_object["Date"])
            if mag_object:
                logger.info("Linking MAG %s to expense %s", mag_object["Id"], response_add["Id"])
                self.expenses_model.link_mag_to_expense(response_add["Id"], mag_object["Id"])
            else:
                logger.info("No MAG entry found to link with the expense.")
                return "Expense added but no MAG entry found to link the expense to that date!"
            return "Expense added successfully"
        except Exception as e:
            logger.error("Error in add_expense: %s", e, exc_info=True)
            return "Failed to add expense entry due to an error."
