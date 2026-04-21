import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import yfinance as yf

from bujo.models.portfolio_transactions import PortfolioTransactions
from bujo.base import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are a portfolio transactions assistant. You handle two use cases using two tools.",
    "⚠️ CRITICAL: You MUST always call a tool. NEVER fabricate results. To add a transaction call `Transaction_Creation`. To list transactions call `Transaction_Lookup`.",

    "Use Case 1: When the user wants to **add/record a transaction** (buy or sell), follow these instructions:",
    "• Extract these fields from natural language:",
    "  - Ticker: stock symbol, e.g. 'PFC.NS', 'INFY.NS', 'AAPL'. If user says 'PFC' assume '.NS' suffix for Indian stocks unless context says otherwise.",
    "  - TransactionType: 'Buy' or 'Sell' (capitalised).",
    "  - NoOfShares: number of shares (float).",
    "  - CostPerShare: price per share (float). For INR stocks this is ₹, for USD stocks this is $.",
    "  - Date: in YYYY-MM-DD format. 'today', 'yesterday' etc. should be resolved using today's date.",
    "  - Portfolio: optional portfolio name if user mentions one (e.g. 'in my LT portfolio'). Default to 'Default' if not mentioned.",
    "  - CMP: omit this field — it will be updated by the scheduled job.",
    "• Example inputs → JSON:",
    "  'Sold 100 shares of PFC.NS at 12.9 today' → {{\"Ticker\": \"PFC.NS\", \"TransactionType\": \"Sell\", \"NoOfShares\": 100, \"CostPerShare\": 12.9, \"Date\": \"{today_date}\", \"Portfolio\": \"Default\"}}",
    "  'Bought 50 INFY.NS at 1500 on 2025-03-01 in LT portfolio' → {{\"Ticker\": \"INFY.NS\", \"TransactionType\": \"Buy\", \"NoOfShares\": 50, \"CostPerShare\": 1500, \"Date\": \"2025-03-01\", \"Portfolio\": \"LT\"}}",
    "• Call `Transaction_Creation` with the JSON string.",
    "• On success respond: '✅ Transaction recorded: [Buy/Sell] NoOfShares shares of Ticker at ₹CostPerShare on Date'.",

    "Use Case 2: When the user wants to **list/view transactions**, follow these instructions:",
    "• Always send tool inputs as **JSON strings**.",
    "• Supported filters (NocoDB syntax): date range, ticker, transaction type.",
    "• Example: 'Show my transactions for March 2025' → {{\"filters\": [\"(CreatedAt,ge,exactDate,2025-03-01)\", \"(CreatedAt,lt,exactDate,2025-04-01)\"]}}",
    "• Example: 'Show all sells of PFC.NS' → {{\"filters\": [\"(Ticker,eq,text,PFC.NS)\", \"(TransactionType,eq,text,Sell)\"]}}",
    "• Example: 'Show all transactions today' → {{\"filters\": [\"(CreatedAt,ge,exactDate,{today_date})\", \"(CreatedAt,lt,exactDate,{tomorrow_date})\"]}}",
    "• Example: 'List all transactions' → pass an empty string '' to fetch all.",
    "• Once fetched, summarise clearly grouped by Ticker or date as relevant. Show Ticker, Type, Shares, Cost, Date.",

    "Remember that today's date is {today_date}, and the week starts on Monday.",

    "📌 Always return responses as a **string**, never a raw JSON object.",
    "💰 Display INR amounts with ₹ and USD amounts with $.",
    "🧾 Format outputs cleanly using **markdown**.",
    "✨ Use emojis where appropriate.",
]


class PortfolioManager:
    def __init__(self, transactions_model: PortfolioTransactions):
        self.transactions_model = transactions_model

        tools = [
            Tool(
                name="Transaction_Lookup",
                func=self.transactions_model.list,
                description="Fetch portfolio transactions based on filters.",
            ),
            Tool(
                name="Transaction_Creation",
                func=self.add_transaction,
                description="Create a new portfolio transaction.",
            ),
        ]

        _memory = MemorySaver()

        def _state_modifier(state):
            now = datetime.now()
            today = now.strftime("%Y-%m-%d %A")
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            content = ("\n".join(SYSTEM_PROMPT)
                       .replace("{today_date}", today)
                       .replace("{tomorrow_date}", tomorrow))
            return [SystemMessage(content=content)] + state["messages"]

        self.agent = create_react_agent(llm, tools, prompt=_state_modifier, checkpointer=_memory)
        logger.info("PortfolioManager initialized.")

    def agent_portfolio(self, prompt: str) -> str:
        text = prompt.strip()
        logger.info("Received prompt: %s", text)
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": "portfolio"}},
            )
            output = result["messages"][-1].content
            logger.info("Agent response: %s", output)
            return output
        except Exception as e:
            logger.error("Error in agent_portfolio: %s", e, exc_info=True)
            return "An error occurred while processing your request."

    def add_transaction(self, data: str) -> str:
        logger.info("Adding transaction with data: %s", data)
        try:
            data_object = json.loads(data)
            response_add = self.transactions_model.create(data_object)
            logger.info("Transaction creation response: %s", response_add)
            if "failed" in str(response_add).lower():
                logger.warning("Failed to add transaction entry.")
                return "Failed to add transaction entry. Try again?"
            return "Transaction added successfully"
        except Exception as e:
            logger.error("Error in add_transaction: %s", e, exc_info=True)
            return "Failed to add transaction due to an error."

    def update_cmp(self) -> str:
        """Fetch latest prices from yfinance and update all transaction rows in NocoDB."""
        transactions = self.transactions_model.list()
        tickers = {tx.get("Ticker") for tx in transactions if tx.get("Ticker")}
        if not tickers:
            return "No tickers found in transactions table."
        for ticker in tickers:
            cmp = yf.Ticker(ticker).info.get("currentPrice", 0)
            for tx in transactions:
                if tx.get("Ticker") == ticker:
                    tx["CMP"] = cmp
                    self.transactions_model.update(tx)
            logger.info("Updated CMP for %s: %s", ticker, cmp)
        return "✅ CMP values updated for all tickers."

    def get_profit_loss_report(self) -> str:
        """Compute P&L across all portfolio positions and return a formatted Markdown string.

        CostPerShare and CMP from yfinance are both in the stock's native currency
        (₹ for .NS, USD for US stocks), so both need USD→INR conversion for US-listed stocks.
        """
        transactions = self.transactions_model.list()
        if not transactions:
            return "📭 No portfolio transactions found."

        usd_inr_ticker = yf.Ticker("USDINR=X")
        usd_to_inr = (
            usd_inr_ticker.info.get("regularMarketPrice")
            or usd_inr_ticker.fast_info.get("lastPrice", 84.0)
        )
        logger.info("USD → INR rate: %s", usd_to_inr)

        ticker_data: Dict[tuple, Dict] = {}
        for tx in transactions:
            ticker    = tx.get("Ticker", "").strip()
            tx_type   = tx.get("TransactionType", "").strip()
            shares    = float(tx.get("NoOfShares") or 0)
            cost      = float(tx.get("CostPerShare") or 0)
            cmp       = float(tx.get("CMP") or 0)
            portfolio = tx.get("Portfolio", "Unknown").strip()

            if not ticker or shares == 0:
                continue

            is_inr   = ticker.upper().endswith(".NS")
            currency = "INR" if is_inr else "USD"
            fx       = 1.0 if is_inr else usd_to_inr

            key = (ticker, portfolio)
            if key not in ticker_data:
                ticker_data[key] = {
                    "portfolio": portfolio, "currency": currency,
                    "total_bought": 0.0, "total_sold": 0.0,
                    "buy_cost_inr": 0.0, "sell_proceeds_inr": 0.0,
                    "cmp_inr": cmp * fx,
                }
            if cmp:
                ticker_data[key]["cmp_inr"] = cmp * fx
            if tx_type == "Buy":
                ticker_data[key]["total_bought"]       += shares
                ticker_data[key]["buy_cost_inr"]       += shares * cost * fx
            elif tx_type == "Sell":
                ticker_data[key]["total_sold"]          += shares
                ticker_data[key]["sell_proceeds_inr"]   += shares * cost * fx

        open_tickers:   List[Dict] = []
        closed_tickers: List[Dict] = []

        for (ticker, __), d in ticker_data.items():
            net_shares   = d["total_bought"] - d["total_sold"]
            avg_cost_inr = (d["buy_cost_inr"] / d["total_bought"]) if d["total_bought"] else 0.0
            cmp_inr      = d["cmp_inr"]

            if net_shares > 0:
                current_value_inr  = net_shares * cmp_inr
                invested_value_inr = net_shares * avg_cost_inr
                unrealised_pl      = current_value_inr - invested_value_inr
                unrealised_pct     = (unrealised_pl / invested_value_inr * 100) if invested_value_inr else 0.0
                realised_pl        = d["sell_proceeds_inr"] - (d["total_sold"] * avg_cost_inr)
                open_tickers.append({
                    "ticker": ticker, "currency": d["currency"],
                    "net_shares": net_shares, "avg_cost_inr": avg_cost_inr,
                    "cmp_inr": cmp_inr, "invested_inr": invested_value_inr,
                    "current_inr": current_value_inr, "unrealised_pl": unrealised_pl,
                    "unrealised_pct": unrealised_pct, "realised_pl": realised_pl,
                    "portfolio": d["portfolio"],
                })
            else:
                realised_pl  = d["sell_proceeds_inr"] - d["buy_cost_inr"]
                realised_pct = (realised_pl / d["buy_cost_inr"] * 100) if d["buy_cost_inr"] else 0.0
                closed_tickers.append({
                    "ticker": ticker, "currency": d["currency"],
                    "buy_cost_inr": d["buy_cost_inr"], "sell_inr": d["sell_proceeds_inr"],
                    "realised_pl": realised_pl, "realised_pct": realised_pct,
                    "portfolio": d["portfolio"],
                })

        def fmt_inr(v: float) -> str: return f"₹{v:,.2f}"
        def pl_emoji(v: float) -> str: return "🟢" if v >= 0 else "🔴"

        def build_section(section_open, section_closed):
            """Build report lines for a group of tickers. All values are in ₹."""
            sec: List[str] = []

            if section_open:
                inv_total  = sum(t["invested_inr"] for t in section_open)
                cur_total  = sum(t["current_inr"]  for t in section_open)
                unreal_tot = cur_total - inv_total
                real_open  = sum(t["realised_pl"]  for t in section_open)
                sec.append(f"📂 *Open* ({len(section_open)})")
                sec.append("─────────────────────────────")
                for t in sorted(section_open, key=lambda x: x["unrealised_pl"], reverse=True):
                    sec.append(
                        f"{pl_emoji(t['unrealised_pl'])} *{t['ticker']}*\n"
                        f"   Shares: `{t['net_shares']:.4f}` | Avg Cost: {fmt_inr(t['avg_cost_inr'])}\n"
                        f"   CMP: {fmt_inr(t['cmp_inr'])} | Invested: {fmt_inr(t['invested_inr'])}\n"
                        f"   Current: {fmt_inr(t['current_inr'])} | "
                        f"Unrealised: {fmt_inr(t['unrealised_pl'])} ({t['unrealised_pct']:+.2f}%)"
                        + (f"\n   Realised (partial): {fmt_inr(t['realised_pl'])}" if t["realised_pl"] != 0 else "")
                    )
                sec.append("─────────────────────────────")
                sec.append(
                    f"📌 *Open Totals*\n"
                    f"   Invested: {fmt_inr(inv_total)}\n"
                    f"   Current:  {fmt_inr(cur_total)}\n"
                    f"   {pl_emoji(unreal_tot)} Unrealised P&L: {fmt_inr(unreal_tot)} "
                    f"({(unreal_tot / inv_total * 100) if inv_total else 0:+.2f}%)"
                    + (f"\n   Realised (partial sells): {fmt_inr(real_open)}" if real_open else "")
                )
            else:
                sec.append("📂 *Open Positions:* None")

            if section_closed:
                buy_tot  = sum(t["buy_cost_inr"] for t in section_closed)
                sell_tot = sum(t["sell_inr"]      for t in section_closed)
                real_tot = sell_tot - buy_tot
                sec.append("")
                sec.append(f"✅ *Closed* ({len(section_closed)})")
                sec.append("─────────────────────────────")
                for t in sorted(section_closed, key=lambda x: x["realised_pl"], reverse=True):
                    sec.append(
                        f"{pl_emoji(t['realised_pl'])} *{t['ticker']}*\n"
                        f"   Invested: {fmt_inr(t['buy_cost_inr'])} | Sold: {fmt_inr(t['sell_inr'])}\n"
                        f"   Realised P&L: {fmt_inr(t['realised_pl'])} ({t['realised_pct']:+.2f}%)"
                    )
                sec.append("─────────────────────────────")
                sec.append(
                    f"📌 *Closed Totals*\n"
                    f"   Total Invested: {fmt_inr(buy_tot)}\n"
                    f"   Total Sold:     {fmt_inr(sell_tot)}\n"
                    f"   {pl_emoji(real_tot)} Realised P&L: {fmt_inr(real_tot)} "
                    f"({(real_tot / buy_tot * 100) if buy_tot else 0:+.2f}%)"
                )
            else:
                sec.append("\n✅ *Closed Positions:* None")

            return sec

        # Group tickers by portfolio
        portfolios: Dict[str, Dict[str, List]] = {}
        for t in open_tickers:
            portfolios.setdefault(t["portfolio"], {"open": [], "closed": []})["open"].append(t)
        for t in closed_tickers:
            portfolios.setdefault(t["portfolio"], {"open": [], "closed": []})["closed"].append(t)

        lines: List[str] = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 *Portfolio P&L Summary*",
            f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            f"💱 USD → INR (CMP): ₹{usd_to_inr:.2f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        grand_invested = grand_current = 0.0

        for portfolio_name in sorted(portfolios):
            pd = portfolios[portfolio_name]
            lines.append("")
            lines.append(f"💼 *{portfolio_name}*")

            port_invested = port_current = 0.0

            inr_open   = [t for t in pd["open"]   if t["currency"] == "INR"]
            inr_closed = [t for t in pd["closed"] if t["currency"] == "INR"]
            usd_open   = [t for t in pd["open"]   if t["currency"] == "USD"]
            usd_closed = [t for t in pd["closed"] if t["currency"] == "USD"]

            if inr_open or inr_closed:
                lines.append("🇮🇳 *Indian Stocks*")
                lines.extend(build_section(inr_open, inr_closed))
                inr_inv = sum(t["invested_inr"] for t in inr_open) + sum(t["buy_cost_inr"] for t in inr_closed)
                inr_cur = sum(t["current_inr"]  for t in inr_open) + sum(t["sell_inr"]     for t in inr_closed)
                inr_pl  = inr_cur - inr_inv
                port_invested += inr_inv; port_current += inr_cur
                grand_invested += inr_inv; grand_current += inr_cur
                lines.append(f"   🏦 Indian Subtotal: {fmt_inr(inr_inv)} → {fmt_inr(inr_cur)}  {pl_emoji(inr_pl)} {fmt_inr(inr_pl)}")

            if usd_open or usd_closed:
                lines.append("🇺🇸 *US Stocks (cost in ₹, CMP converted from USD)*")
                lines.extend(build_section(usd_open, usd_closed))
                usd_inv = sum(t["invested_inr"] for t in usd_open) + sum(t["buy_cost_inr"] for t in usd_closed)
                usd_cur = sum(t["current_inr"]  for t in usd_open) + sum(t["sell_inr"]     for t in usd_closed)
                usd_pl  = usd_cur - usd_inv
                port_invested += usd_inv; port_current += usd_cur
                grand_invested += usd_inv; grand_current += usd_cur
                lines.append(f"   🏦 US Subtotal: {fmt_inr(usd_inv)} → {fmt_inr(usd_cur)}  {pl_emoji(usd_pl)} {fmt_inr(usd_pl)}")

            port_pl = port_current - port_invested
            lines.append("─────────────────────────────")
            lines.append(
                f"📌 *{portfolio_name} Total*: {fmt_inr(port_invested)} → {fmt_inr(port_current)}  "
                f"{pl_emoji(port_pl)} {fmt_inr(port_pl)} "
                f"({(port_pl / port_invested * 100) if port_invested else 0:+.2f}%)"
            )

        grand_pl = grand_current - grand_invested
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💼 *Grand Total (INR equivalent)*\n"
            f"   Total Invested: {fmt_inr(grand_invested)}\n"
            f"   Total Value:    {fmt_inr(grand_current)}\n"
            f"   {pl_emoji(grand_pl)} Overall P&L: {fmt_inr(grand_pl)} "
            f"({(grand_pl / grand_invested * 100) if grand_invested else 0:+.2f}%)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        return "\n".join(lines)
