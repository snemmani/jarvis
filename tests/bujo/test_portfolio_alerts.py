import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("TELEGRAM_USER_ID", "1")
os.environ.setdefault("OPENAI_MODEL", "gpt-test")
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
fake_base.OPENAI_MODEL = "gpt-test"
fake_base.openai_model = MagicMock()
fake_base.llm = MagicMock()
sys.modules["bujo.base"] = fake_base

fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = MagicMock()
sys.modules.setdefault("yfinance", fake_yf)

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

from bujo.portoflio.alerts import _check_thesis_note, _compute_open_positions


def _make_llm_response(items: list) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(items)
    return resp


class TestCheckThesisNote(unittest.TestCase):
    def setUp(self):
        fake_base.openai_model.chat.completions.create.reset_mock(side_effect=True, return_value=True)

    # --- guard clauses ---

    def test_empty_note_returns_no_alerts(self):
        result = _check_thesis_note("Acme Corp", "ACME.NS", "", [{"title": "Acme posts record profits"}])
        self.assertEqual(result, [])
        fake_base.openai_model.chat.completions.create.assert_not_called()

    def test_whitespace_only_note_returns_no_alerts(self):
        result = _check_thesis_note("Acme Corp", "ACME.NS", "   ", [])
        self.assertEqual(result, [])
        fake_base.openai_model.chat.completions.create.assert_not_called()

    # --- LLM returns nothing triggered ---

    def test_llm_returns_empty_array_produces_no_alerts(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([])
        result = _check_thesis_note(
            "MRF Ltd", "MRF.NS",
            "Exit on monsoon shock + tractor demand collapse.",
            [{"title": "MRF reports steady Q4 volumes"}],
        )
        self.assertEqual(result, [])

    # --- LLM returns triggered conditions ---

    def test_single_triggered_condition_formats_alert(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([
            {"condition": "exit on RBI restrictive action", "evidence": "RBI issued circular restricting NBFC lending limits"}
        ])
        result = _check_thesis_note(
            "Shriram Finance", "SHRIRAMFIN.NS",
            "Trim if NIM falls below 9.0%; exit on RBI restrictive action.",
            [{"title": "RBI tightens NBFC lending norms"}],
        )
        self.assertEqual(len(result), 1)
        self.assertIn("📋 *Thesis alert*", result[0])
        self.assertIn("exit on RBI restrictive action", result[0])
        self.assertIn("RBI issued circular", result[0])

    def test_multiple_triggered_conditions_all_returned(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([
            {"condition": "Trim if order inflow growth <10%", "evidence": "Q4 order book grew only 7% YoY per management commentary"},
            {"condition": "exit if working capital exceeds 12% of sales", "evidence": "Working capital at 14% as per latest balance sheet"},
        ])
        result = _check_thesis_note(
            "L&T", "LT.NS",
            "Trim if order inflow growth <10% for two consecutive quarters; exit if working capital exceeds 12% of sales.",
            [{"title": "L&T Q4 results — order inflows disappoint"}, {"title": "L&T working capital rises sharply"}],
        )
        self.assertEqual(len(result), 2)

    # --- LLM response robustness ---

    def test_llm_response_with_missing_evidence_field_still_returns_alert(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([
            {"condition": "exit on governance shock"}
        ])
        result = _check_thesis_note(
            "Zee Entertainment", "ZEEL.NS",
            "Exit on governance shock.",
            [{"title": "Zee promoter arrested"}],
        )
        self.assertEqual(len(result), 1)
        self.assertIn("exit on governance shock", result[0])

    def test_llm_returns_item_without_condition_field_is_skipped(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([
            {"evidence": "some evidence but no condition key"}
        ])
        result = _check_thesis_note(
            "Tata Motors", "TATAMOTORS.NS",
            "Trim if SUV market share drops 200 bps.",
            [{"title": "Tata Motors loses ground to Mahindra"}],
        )
        self.assertEqual(result, [])

    def test_llm_returns_malformed_json_returns_no_alerts(self):
        resp = MagicMock()
        resp.choices[0].message.content = "Sorry, I cannot evaluate this."
        fake_base.openai_model.chat.completions.create.return_value = resp
        result = _check_thesis_note(
            "HDFC Bank", "HDFCBANK.NS",
            "Trim if NIM falls below 9.0%.",
            [{"title": "HDFC Bank Q3 NIM at 8.8%"}],
        )
        self.assertEqual(result, [])

    def test_llm_raises_exception_returns_no_alerts(self):
        fake_base.openai_model.chat.completions.create.side_effect = RuntimeError("API timeout")
        result = _check_thesis_note(
            "Infosys", "INFY.NS",
            "Exit on governance shock or major client loss.",
            [{"title": "Infosys loses major US banking client"}],
        )
        self.assertEqual(result, [])

    # --- no news case ---

    def test_no_news_items_still_calls_llm_with_placeholder(self):
        fake_base.openai_model.chat.completions.create.return_value = _make_llm_response([])
        _check_thesis_note(
            "Mahindra & Mahindra", "M&M.NS",
            "Trim if SUV market share drops 200 bps.",
            [],
        )
        call_args = fake_base.openai_model.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        self.assertIn("No recent news headlines available.", prompt)


class TestComputeOpenPositionsNote(unittest.TestCase):

    def test_note_collected_from_buy_transaction(self):
        txs = [
            {
                "Ticker": "INFY.NS", "TransactionType": "Buy",
                "NoOfShares": 10, "CostPerShare": 1500, "CMP": 1600,
                "Date": "2025-01-01",
                "Note": "Exit on governance shock or major client loss.",
            }
        ]
        positions = _compute_open_positions(txs)
        self.assertEqual(positions["INFY.NS"]["note"], "Exit on governance shock or major client loss.")

    def test_empty_note_stored_as_empty_string(self):
        txs = [
            {
                "Ticker": "TCS.NS", "TransactionType": "Buy",
                "NoOfShares": 5, "CostPerShare": 3000, "CMP": 3200,
                "Date": "2025-01-01",
                "Note": "",
            }
        ]
        positions = _compute_open_positions(txs)
        self.assertEqual(positions["TCS.NS"]["note"], "")

    def test_missing_note_key_stored_as_empty_string(self):
        txs = [
            {
                "Ticker": "WIPRO.NS", "TransactionType": "Buy",
                "NoOfShares": 20, "CostPerShare": 400, "CMP": 450,
                "Date": "2025-01-01",
            }
        ]
        positions = _compute_open_positions(txs)
        self.assertEqual(positions["WIPRO.NS"]["note"], "")

    def test_last_non_empty_note_wins_across_multiple_transactions(self):
        txs = [
            {
                "Ticker": "HDFCBANK.NS", "TransactionType": "Buy",
                "NoOfShares": 10, "CostPerShare": 1400, "CMP": 1500,
                "Date": "2024-06-01", "Note": "First thesis note.",
            },
            {
                "Ticker": "HDFCBANK.NS", "TransactionType": "Buy",
                "NoOfShares": 5, "CostPerShare": 1450, "CMP": 1500,
                "Date": "2024-09-01", "Note": "Updated thesis: trim if NIM < 9%.",
            },
        ]
        positions = _compute_open_positions(txs)
        self.assertEqual(positions["HDFCBANK.NS"]["note"], "Updated thesis: trim if NIM < 9%.")

    def test_sell_transaction_note_ignored_when_earlier_buy_note_exists(self):
        txs = [
            {
                "Ticker": "RELIANCE.NS", "TransactionType": "Buy",
                "NoOfShares": 10, "CostPerShare": 2000, "CMP": 2200,
                "Date": "2024-01-01", "Note": "Hold for jio spinoff.",
            },
            {
                "Ticker": "RELIANCE.NS", "TransactionType": "Sell",
                "NoOfShares": 3, "CostPerShare": 2200, "CMP": 2200,
                "Date": "2024-06-01", "Note": "",
            },
        ]
        positions = _compute_open_positions(txs)
        self.assertEqual(positions["RELIANCE.NS"]["note"], "Hold for jio spinoff.")


if __name__ == "__main__":
    unittest.main()
