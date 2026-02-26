import logging
from datetime import datetime
from typing import List, Dict
import json

from bujo.models.portfolio_transactions import PortfolioTransactions
from bujo.models.mag import MAG
from bujo.base import llm
from langchain_core.tools import Tool
from langchain_classic.agents import initialize_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.prompts import ChatPromptTemplate


SYSTEM_PROMPT = [
    "You are a portfolio transactions assistant. Use tools to list and create portfolio transactions.",
    "When creating transactions, accept a JSON string and call the creation tool with that string.",
    "When listing transactions, accept date-based filters as a JSON string and call the lookup tool with that string."
]


class PortfolioManager:
    def __init__(self, transactions_model: PortfolioTransactions, mag_model: MAG):
        """
        Manager for portfolio transactions.

        :param transactions_model: Instance of `PortfolioTransactions` model
        :param mag_model: Instance of `MAG` model
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.transactions_model = transactions_model
        self.mag_model = mag_model

        self.tools = [
            Tool(
                name="Transaction Lookup",
                func=self.transactions_model.list,
                description="Fetch portfolio transactions based on filters."
            ),
            Tool(
                name="Transaction Creation",
                func=self.add_transaction,
                description="Create a new portfolio transaction."
            )
        ]

        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=1)

        self.prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": '\n'.join(SYSTEM_PROMPT)},
            {"role": "human", "content": "{input}"}
        ])

        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def agent_portfolio(self, prompt: str):
        text = prompt.strip()
        self.logger.info(f"Received prompt: {text}")
        try:
            response = self.agent.invoke(str(self.prompt.format_messages(input=text)))
            self.logger.info(f"Agent response: {response}")
            return response.get('output')
        except Exception as e:
            self.logger.error(f"Error in agent_portfolio: {e}", exc_info=True)
            return "An error occurred while processing your request."

    def add_transaction(self, data: str):
        """
        Create a portfolio transaction.

        :param data: JSON string containing the transaction data.
        :return: Result message or created record.
        """
        self.logger.info(f"Adding transaction with data: {data}")
        try:
            data_object = json.loads(data)
            response_add = self.transactions_model.create(data_object)
            self.logger.info(f"Transaction creation response: {response_add}")
            if 'failed' in str(response_add).lower():
                self.logger.warning("Failed to add transaction entry.")
                return "Failed to add transaction entry. Try again?"

            # Attempt to link to MAG by Date if present
            date_val = data_object.get('Date')
            if date_val:
                mag_object = self.mag_model.find_by_date(date_val)
                if mag_object:
                    try:
                        self.transactions_model.link_mag_to_transaction(response_add['Id'], mag_object['Id'])
                        return "Transaction added and linked to MAG successfully"
                    except Exception:
                        self.logger.info("Transaction added but linking to MAG failed.")
                        return "Transaction added but failed to link to MAG."

            return "Transaction added successfully"
        except Exception as e:
            self.logger.error(f"Error in add_transaction: {e}", exc_info=True)
            return "Failed to add transaction due to an error."
