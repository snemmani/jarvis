import unittest

from bujo.portoflio.ledger import (
    build_portfolio_ledger,
    compute_cash_by_portfolio,
    compute_positions_by_portfolio,
)


class TestPortfolioLedger(unittest.TestCase):
    def test_partial_sell_keeps_open_basis_and_realised_pl(self):
        ledger = build_portfolio_ledger([
            {"Ticker": "INFY.NS", "TransactionType": "Buy", "NoOfShares": 10, "CostPerShare": 1000, "CMP": 1400, "Portfolio": "Core"},
            {"Ticker": "INFY.NS", "TransactionType": "Sell", "NoOfShares": 4, "CostPerShare": 1300, "CMP": 1400, "Portfolio": "Core"},
        ])

        holding = ledger["portfolios"]["Core"]["holdings"][0]

        self.assertEqual(holding["net_shares"], 6)
        self.assertEqual(holding["avg_cost_inr"], 1000)
        self.assertEqual(holding["invested_inr"], 6000)
        self.assertEqual(holding["current_inr"], 8400)
        self.assertEqual(holding["unrealised_pl"], 2400)
        self.assertEqual(holding["realised_pl"], 1200)

    def test_closed_position_tracks_realised_pl(self):
        ledger = build_portfolio_ledger([
            {"Ticker": "TCS.NS", "TransactionType": "Buy", "NoOfShares": 2, "CostPerShare": 3000, "Portfolio": "Core"},
            {"Ticker": "TCS.NS", "TransactionType": "Sell", "NoOfShares": 2, "CostPerShare": 3500, "Portfolio": "Core"},
        ])

        closed = ledger["portfolios"]["Core"]["closed_positions"][0]

        self.assertEqual(ledger["portfolios"]["Core"]["holdings"], [])
        self.assertEqual(closed["buy_cost_inr"], 6000)
        self.assertEqual(closed["sell_inr"], 7000)
        self.assertEqual(closed["realised_pl"], 1000)

    def test_closed_lot_does_not_pollute_reentry_average_cost(self):
        ledger = build_portfolio_ledger([
            {"Id": 1, "Ticker": "TCS.NS", "Date": "2025-10-31", "TransactionType": "Sell", "NoOfShares": 10, "CostPerShare": 3061.30, "CMP": 2151, "Portfolio": "Ishaan"},
            {"Id": 2, "Ticker": "TCS.NS", "Date": "2024-06-10", "TransactionType": "Buy", "NoOfShares": 10, "CostPerShare": 3860, "CMP": 2151, "Portfolio": "Ishaan"},
            {"Id": 3, "Ticker": "TCS.NS", "Date": "2026-06-09", "TransactionType": "Buy", "NoOfShares": 6, "CostPerShare": 2135.30, "CMP": 2151, "Portfolio": "Ishaan"},
        ])

        holding = ledger["portfolios"]["Ishaan"]["holdings"][0]

        self.assertEqual(holding["net_shares"], 6)
        self.assertAlmostEqual(holding["avg_cost_inr"], 2135.30)
        self.assertAlmostEqual(holding["invested_inr"], 12811.80)
        self.assertAlmostEqual(holding["unrealised_pct"], 0.735256, places=5)

    def test_cash_rows_use_explicit_cash_ledger_only(self):
        ledger = build_portfolio_ledger([
            {"Ticker": "CASH", "TransactionType": "Deposit", "NoOfShares": 1, "CostPerShare": 10000, "Portfolio": "Core"},
            {"Ticker": "INFY.NS", "TransactionType": "Buy", "NoOfShares": 1, "CostPerShare": 1000, "CMP": 1100, "Portfolio": "Core"},
        ])

        self.assertEqual(ledger["portfolios"]["Core"]["cash_inr"], 10000)
        self.assertEqual(ledger["totals"]["total_value_inr"], 11100)

    def test_us_stock_uses_supplied_fx_for_cost_and_cmp(self):
        ledger = build_portfolio_ledger([
            {"Ticker": "AAPL", "TransactionType": "Buy", "NoOfShares": 2, "CostPerShare": 100, "CMP": 125, "Portfolio": "Core"},
        ], fx_rates={"USD": 80})

        holding = ledger["portfolios"]["Core"]["holdings"][0]

        self.assertEqual(holding["avg_cost_inr"], 8000)
        self.assertEqual(holding["cmp_inr"], 10000)
        self.assertEqual(holding["invested_inr"], 16000)
        self.assertEqual(holding["current_inr"], 20000)

    def test_oversell_generates_warning_without_negative_holding(self):
        ledger = build_portfolio_ledger([
            {"Ticker": "SBIN.NS", "TransactionType": "Buy", "NoOfShares": 2, "CostPerShare": 500, "Portfolio": "Core"},
            {"Ticker": "SBIN.NS", "TransactionType": "Sell", "NoOfShares": 3, "CostPerShare": 550, "Portfolio": "Core"},
        ])

        self.assertEqual(ledger["portfolios"]["Core"]["holdings"], [])
        self.assertTrue(any("Oversold SBIN.NS" in warning for warning in ledger["warnings"]))

    def test_rebalance_helper_uses_only_open_lots_after_reentry(self):
        positions = compute_positions_by_portfolio([
            {"Id": 1, "Ticker": "TCS.NS", "Date": "2025-10-31", "TransactionType": "Sell", "NoOfShares": 10, "CostPerShare": 3061.30, "Portfolio": "Ishaan"},
            {"Id": 2, "Ticker": "TCS.NS", "Date": "2024-06-10", "TransactionType": "Buy", "NoOfShares": 10, "CostPerShare": 3860, "Portfolio": "Ishaan"},
            {"Id": 3, "Ticker": "TCS.NS", "Date": "2026-06-09", "TransactionType": "Buy", "NoOfShares": 6, "CostPerShare": 2135.30, "Portfolio": "Ishaan"},
        ])

        pos = positions["Ishaan"]["TCS.NS"]

        self.assertEqual(pos["net_shares"], 6)
        self.assertAlmostEqual(pos["avg_cost"], 2135.30)
        self.assertAlmostEqual(pos["total_invested"], 12811.80)
        self.assertEqual(pos["buy_dates"], ["2026-06-09"])
        self.assertEqual(pos["all_buy_lots"], [("2026-06-09", 6, 2135.30)])

    def test_rebalance_helpers_share_normalisation(self):
        txs = [
            {"Ticker": "cash", "TransactionType": "Deposit", "NoOfShares": 1, "CostPerShare": 5000, "Portfolio": "Core"},
            {"Ticker": "infy.ns", "TransactionType": "buy", "NoOfShares": "3", "CostPerShare": "1,000", "Date": "2026-01-02", "Portfolio": "Core", "Note": "Thesis"},
            {"Ticker": "INFY.NS", "TransactionType": "Sell", "NoOfShares": 1, "CostPerShare": 1100, "Date": "2026-01-03", "Portfolio": "Core"},
        ]

        cash = compute_cash_by_portfolio(txs)
        positions = compute_positions_by_portfolio(txs)

        self.assertEqual(cash["Core"], 5000)
        self.assertEqual(positions["Core"]["INFY.NS"]["net_shares"], 2)
        self.assertEqual(positions["Core"]["INFY.NS"]["avg_cost"], 1000)
        self.assertEqual(positions["Core"]["INFY.NS"]["notes"][0]["note"], "Thesis")


if __name__ == "__main__":
    unittest.main()
