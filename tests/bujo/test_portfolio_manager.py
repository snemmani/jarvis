import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


os.environ.setdefault("TELEGRAM_USER_ID", "1")
os.environ.setdefault("OPENAI_MODEL", "gpt-5-mini")
os.environ.setdefault("TEXT_TO_SPEECH_MODEL", "tts-1")
os.environ.setdefault("NOCODB_BASE_URL", "https://example.com")
os.environ.setdefault("NOCODB_API_TOKEN", "token")
os.environ.setdefault("NOCODB_EXPENSES_TABLE_ID", "expenses")
os.environ.setdefault("NOCODB_TRANSACTIONS_TABLE_ID", "transactions")
os.environ.setdefault("NOCODB_PRICE_ALERTS_TABLE_ID", "alerts")
os.environ.setdefault("NOCODB_MAG_TABLE_ID", "mag")
os.environ.setdefault("NOCODB_EXPENSES_MAG_LINK_ID", "link")
os.environ.setdefault("TELEGRAM_TOKEN", "token")
os.environ.setdefault("WOLFRAM_APP_ID", "app")
os.environ.setdefault("PC_MAC_ADDRESS", "00:00:00:00:00:00")
os.environ.setdefault("BROADCAST_IP", "127.0.0.1")
os.environ.setdefault("CHAT_ID", "1")
os.environ.setdefault("NOIP_USERNAME", "user")
os.environ.setdefault("NOIP_PASSWORD", "pass")
os.environ.setdefault("NOIP_HOSTNAME", "host")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

fake_base = types.ModuleType("bujo.base")
fake_base.llm = object()
sys.modules.setdefault("bujo.base", fake_base)

fake_yfinance = types.ModuleType("yfinance")
fake_yfinance.Ticker = MagicMock()
sys.modules.setdefault("yfinance", fake_yfinance)

fake_messages = types.ModuleType("langchain_core.messages")
fake_messages.HumanMessage = object
fake_messages.SystemMessage = object
sys.modules.setdefault("langchain_core.messages", fake_messages)

fake_tools = types.ModuleType("langchain_core.tools")
fake_tools.Tool = MagicMock()
sys.modules.setdefault("langchain_core.tools", fake_tools)

fake_memory = types.ModuleType("langgraph.checkpoint.memory")
fake_memory.MemorySaver = MagicMock()
sys.modules.setdefault("langgraph.checkpoint.memory", fake_memory)

fake_prebuilt = types.ModuleType("langgraph.prebuilt")
fake_prebuilt.create_react_agent = MagicMock()
sys.modules.setdefault("langgraph.prebuilt", fake_prebuilt)

from bujo.models.portfolio_transactions import PortfolioTransactions
from bujo.portoflio.manage import PortfolioManager



class TestPortfolioTransactionsModel(unittest.TestCase):
    @patch("requests.get")
    def test_list_joins_date_range_filters_with_nocodb_and(self, mock_get):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"list": []}
        mock_get.return_value = mock_response
        model = PortfolioTransactions("https://example.com", "token", "transactions")

        model.list(json.dumps({
            "filters": [
                "(Date,ge,exactDate,2026-06-09)",
                "(Date,lt,exactDate,2026-06-10)",
            ]
        }))

        mock_get.assert_called_once_with(
            "https://example.com/api/v2/tables/transactions/records",
            headers=model.headers,
            params={
                "where": "(Date,ge,exactDate,2026-06-09)~and(Date,lt,exactDate,2026-06-10)",
                "limit": 1000,
                "offset": 0,
            },
        )


class TestPortfolioManager(unittest.TestCase):
    @patch.object(PortfolioManager, "_build_agent", autospec=True)
    def test_add_transaction_creates_cash_for_lowercase_buy(self, _mock_build_agent):
        transactions_model = MagicMock()
        transactions_model.create.side_effect = [{"Id": 1}, {"Id": 2}]
        manager = PortfolioManager(transactions_model)

        result = manager.add_transaction(json.dumps({
            "Ticker": "infy.ns",
            "TransactionType": "buy",
            "NoOfShares": "10",
            "CostPerShare": "1500",
            "Date": "2026-05-07",
            "Portfolio": "Default",
        }))

        self.assertEqual(result, "Transaction added successfully")
        self.assertEqual(transactions_model.create.call_count, 2)
        stock_call = transactions_model.create.call_args_list[0].args[0]
        cash_call = transactions_model.create.call_args_list[1].args[0]
        self.assertEqual(stock_call["Ticker"], "INFY.NS")
        self.assertEqual(stock_call["TransactionType"], "Buy")
        self.assertEqual(cash_call["Ticker"], "CASH")
        self.assertEqual(cash_call["TransactionType"], "Withdraw")
        self.assertEqual(cash_call["CostPerShare"], 15000.0)

    @patch.object(PortfolioManager, "_build_agent", autospec=True)
    def test_add_transaction_parses_formatted_numbers_before_cash_entry(self, _mock_build_agent):
        transactions_model = MagicMock()
        transactions_model.create.side_effect = [{"Id": 1}, {"Id": 2}]
        manager = PortfolioManager(transactions_model)

        result = manager.add_transaction(json.dumps({
            "Ticker": "TCS.NS",
            "TransactionType": "Sell",
            "NoOfShares": "1,200",
            "CostPerShare": "₹3,450.50",
            "Date": "2026-05-07",
            "Portfolio": "LT",
        }))

        self.assertEqual(result, "Transaction added successfully")
        cash_call = transactions_model.create.call_args_list[1].args[0]
        self.assertEqual(cash_call["TransactionType"], "Deposit")
        self.assertEqual(cash_call["CostPerShare"], 4140600.0)

    @patch.object(PortfolioManager, "_build_agent", autospec=True)
    @patch.object(PortfolioManager, "_get_usd_to_inr", autospec=True, return_value=80.0)
    def test_dashboard_data_aggregates_cash_and_holdings(self, _mock_fx, _mock_build_agent):
        transactions_model = MagicMock()
        transactions_model.list.return_value = [
            {"Ticker": "CASH", "TransactionType": "Deposit", "NoOfShares": 1, "CostPerShare": 100000, "Portfolio": "Core"},
            {"Ticker": "INFY.NS", "TransactionType": "Buy", "NoOfShares": 10, "CostPerShare": 1500, "CMP": 1600, "Portfolio": "Core"},
            {"Ticker": "AAPL", "TransactionType": "Buy", "NoOfShares": 2, "CostPerShare": 100, "CMP": 120, "Portfolio": "Core"},
            {"Ticker": "CASH", "TransactionType": "Withdraw", "NoOfShares": 1, "CostPerShare": 15000, "Portfolio": "Core"},
        ]
        manager = PortfolioManager(transactions_model)

        dashboard = manager.get_dashboard_data()

        self.assertEqual(dashboard["totals"]["portfolio_count"], 1)
        self.assertEqual(dashboard["totals"]["cash_inr"], 85000.0)
        self.assertEqual(dashboard["totals"]["current_inr"], 35200.0)
        self.assertEqual(dashboard["totals"]["total_value_inr"], 120200.0)
        self.assertEqual(dashboard["holdings"][0]["ticker"], "AAPL")
        self.assertEqual(dashboard["portfolios"]["Core"]["cash_inr"], 85000.0)


if __name__ == "__main__":
    unittest.main()
