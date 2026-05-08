import sys
import types
import unittest
from unittest.mock import MagicMock


fake_base = types.ModuleType("bujo.base")
fake_base.openai_model = object()
fake_base.llm = object()
sys.modules["bujo.base"] = fake_base

fake_yfinance = types.ModuleType("yfinance")
fake_yfinance.Ticker = object
sys.modules["yfinance"] = fake_yfinance

fake_messages = types.ModuleType("langchain_core.messages")
fake_messages.HumanMessage = object
fake_messages.SystemMessage = object
sys.modules["langchain_core.messages"] = fake_messages

fake_tools = types.ModuleType("langchain_core.tools")
fake_tools.Tool = MagicMock()
sys.modules["langchain_core.tools"] = fake_tools

fake_memory = types.ModuleType("langgraph.checkpoint.memory")
fake_memory.MemorySaver = MagicMock()
sys.modules["langgraph.checkpoint.memory"] = fake_memory

fake_prebuilt = types.ModuleType("langgraph.prebuilt")
fake_prebuilt.create_react_agent = MagicMock()
sys.modules["langgraph.prebuilt"] = fake_prebuilt

from bujo.portoflio.rebalance import (
    _compute_valuation_profile,
    _select_valuation_method,
    _summarize_forward_outlook,
)


class TestRebalanceForwardOutlook(unittest.TestCase):
    def test_classifies_high_compounding_setup(self):
        td = {
            "revenue_growth_yoy": 18.0,
            "earnings_growth_yoy": 22.0,
            "eps_ttm": 100.0,
            "eps_forward": 118.0,
            "pe_ttm": 24.0,
            "pe_forward": 21.0,
            "peg": 1.2,
            "roe": 19.0,
            "operating_margin": 21.0,
            "cash_conversion_ratio": 1.05,
            "net_debt_to_ebitda": 0.8,
            "forensic_flags": [],
            "screener_data": {"pros": ["Strong order book"], "cons": []},
        }

        result = _summarize_forward_outlook(
            "TEST",
            td,
            recent_notes=["Trim only if thesis weakens materially"],
            headlines=["Company wins large order book expansion contract"],
            gate_assessment={"status": "eligible", "all_reasons": []},
        )

        self.assertEqual(result["outlook"], "High Compounding Potential")
        self.assertEqual(result["xirr_band"], ">20%")
        self.assertGreaterEqual(result["score"], 5)

    def test_classifies_weaker_setup_with_cautions(self):
        td = {
            "revenue_growth_yoy": -3.0,
            "earnings_growth_yoy": -12.0,
            "eps_ttm": 100.0,
            "eps_forward": 82.0,
            "pe_ttm": 18.0,
            "pe_forward": 24.0,
            "peg": 3.2,
            "roe": 8.0,
            "operating_margin": 6.0,
            "cash_conversion_ratio": 0.62,
            "net_debt_to_ebitda": 3.8,
            "forensic_flags": ["flag1", "flag2"],
            "screener_data": {"pros": [], "cons": ["Margin pressure"]},
        }

        result = _summarize_forward_outlook(
            "TEST",
            td,
            recent_notes=["Exit if governance warning escalates"],
            headlines=["SEBI investigation and margin pressure warning"],
            gate_assessment={"status": "rejected", "all_reasons": ["forward EPS below trailing EPS"]},
        )

        self.assertIn(result["outlook"], {"High Uncertainty", "Low Forward Return Potential"})
        self.assertIn(result["xirr_band"], {"<10%", "10–15%", "Unclear"})
        self.assertLess(result["score"], 2)

    def test_selects_pb_roe_for_financials(self):
        td = {"sector_bucket": "Banks"}
        self.assertEqual(_select_valuation_method(td), "pb_roe")

    def test_builds_dcf_and_reverse_dcf_for_non_financial(self):
        td = {
            "sector_bucket": "Technology",
            "free_cash_flow_cr": 1200.0,
            "shares_outstanding_cr": 240.0,
            "market_cap_cr": 18000.0,
            "cmp": 60.0,
            "beta": 1.0,
            "revenue_growth_yoy": 14.0,
            "earnings_growth_yoy": 16.0,
            "revenue_cagr_3y_pct": 12.0,
            "net_income_cagr_3y_pct": 13.0,
            "roe": 18.0,
            "operating_margin": 19.0,
            "cash_conversion_ratio": 1.0,
            "net_debt_to_ebitda": 1.1,
            "peg": 1.4,
            "forensic_flags": [],
            "screener_data": {"pros": [], "cons": []},
        }
        valuation = _compute_valuation_profile(td)
        self.assertEqual(valuation["primary"]["method"], "DCF")
        self.assertTrue(valuation["reverse"]["applicable"])
        self.assertIn("base", valuation["primary"]["per_share"])

    def test_builds_pb_roe_for_bank(self):
        td = {
            "sector_bucket": "Banks",
            "book_value": 520.0,
            "roe": 16.0,
            "pb": 2.1,
            "cmp": 980.0,
            "beta": 0.9,
            "payout_ratio": 20.0,
        }
        valuation = _compute_valuation_profile(td)
        self.assertEqual(valuation["primary"]["method"], "P/B-ROE")
        self.assertEqual(valuation["reverse"]["method"], "Reverse P/B-ROE")


if __name__ == "__main__":
    unittest.main()
from unittest.mock import MagicMock
