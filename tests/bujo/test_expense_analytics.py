import sys
import types
import unittest
from unittest.mock import MagicMock


fake_base = types.ModuleType("bujo.base")
fake_base.WOLFRAM_APP_ID = "app"
fake_base.expenses_model = MagicMock()
fake_base.llm = MagicMock()
sys.modules["bujo.base"] = fake_base

fake_managers = types.ModuleType("bujo.managers")
fake_managers.expense_manager = MagicMock()
fake_managers.mag_manager = MagicMock()
fake_managers.portfolio_manager = MagicMock()
sys.modules["bujo.managers"] = fake_managers

fake_telegram = types.ModuleType("telegram")
fake_telegram.Update = object
fake_telegram.constants = types.SimpleNamespace(ChatAction=types.SimpleNamespace(TYPING="typing"))
sys.modules.setdefault("telegram", fake_telegram)

fake_telegram_ext = types.ModuleType("telegram.ext")
fake_telegram_ext.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=object)
sys.modules.setdefault("telegram.ext", fake_telegram_ext)

fake_wolfram = types.ModuleType("wolframalpha")
fake_wolfram.Client = MagicMock()
sys.modules.setdefault("wolframalpha", fake_wolfram)

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

from bujo.agent import _apply_expense_filters, _choose_chart_type, _extract_excluded_terms


class TestExpenseAnalyticsFiltering(unittest.TestCase):
    def test_extracts_except_phrase_from_user_text(self):
        terms = _extract_excluded_terms("Get me expenses for month of May except Home Loan", {})

        self.assertEqual(terms, ["Home Loan"])

    def test_excludes_matching_item_before_charting(self):
        expenses = [
            {"Item": "Home Loan", "Amount": 50000, "Date": "2026-05-01"},
            {"Item": "Groceries", "Amount": 3000, "Date": "2026-05-02"},
            {"Item": "Fuel", "Amount": 1500, "Date": "2026-05-03"},
        ]

        filtered, summary = _apply_expense_filters(
            expenses, {}, "Show May expense chart except Home Loan"
        )

        self.assertEqual([e["Item"] for e in filtered], ["Groceries", "Fuel"])
        self.assertIn("excluded Home Loan", summary)
        self.assertIn("3 -> 2", summary)

    def test_exclude_terms_param_overrides_text_parsing(self):
        expenses = [
            {"Item": "Home Loan", "Amount": 50000},
            {"Item": "Dining", "Amount": 1200},
        ]

        filtered, _ = _apply_expense_filters(
            expenses, {"exclude_terms": ["Dining"]}, "May expenses except Home Loan"
        )

        self.assertEqual([e["Item"] for e in filtered], ["Home Loan"])

    def test_chart_type_honors_user_line_chart_request(self):
        chart_type = _choose_chart_type(
            {"chart_type": "pie"},
            "Show May expenses except Home Loan as a line chart",
            "{}",
        )

        self.assertEqual(chart_type, "daily_line")


if __name__ == "__main__":
    unittest.main()
