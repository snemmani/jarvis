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
    "  - Note: optional free-text context if the user mentions thesis, trigger, caution, or any rationale for the transaction.",
    "  - CMP: omit this field — it will be updated by the scheduled job.",
    "• Example inputs → JSON:",
    "  'Sold 100 shares of PFC.NS at 12.9 today' → {{\"Ticker\": \"PFC.NS\", \"TransactionType\": \"Sell\", \"NoOfShares\": 100, \"CostPerShare\": 12.9, \"Date\": \"{today_date}\", \"Portfolio\": \"Default\"}}",
    "  'Bought 50 INFY.NS at 1500 on 2025-03-01 in LT portfolio because results were strong' → {{\"Ticker\": \"INFY.NS\", \"TransactionType\": \"Buy\", \"NoOfShares\": 50, \"CostPerShare\": 1500, \"Date\": \"2025-03-01\", \"Portfolio\": \"LT\", \"Note\": \"because results were strong\"}}",
    "• Call `Transaction_Creation` with the JSON string.",
    "• On success respond: '✅ Transaction recorded: [Buy/Sell] NoOfShares shares of Ticker at ₹CostPerShare on Date'. Mention the note briefly if one was provided.",

    "Use Case 1b: When the user wants to **record a cash deposit or withdrawal** (adding/removing external funds from a portfolio), follow these instructions:",
    "• This is for external cash flows only — fresh capital added or withdrawn from the portfolio. NOT for recording stock trades (buy/sell automatically updates cash).",
    "• Examples: 'deposited 50000 into my LT portfolio', 'added ₹1 lakh cash to Default', 'withdrew 20000 from ST portfolio'.",
    "• Record as: {\"Ticker\": \"CASH\", \"TransactionType\": \"Deposit\" (or \"Withdraw\"), \"NoOfShares\": 1, \"CostPerShare\": <amount>, \"Date\": \"<YYYY-MM-DD>\", \"Portfolio\": \"<portfolio>\", \"Note\": \"<optional context>\"}",
    "• 'deposit'/'add cash'/'fund' → TransactionType = 'Deposit'. 'withdraw'/'remove cash'/'take out' → TransactionType = 'Withdraw'.",
    "• On success respond: '✅ Cash recorded: [Deposit/Withdraw] of ₹<amount> in <portfolio> portfolio on <date>'.",

    "Use Case 2: When the user wants to **list/view transactions**, follow these instructions:",
    "• Always send tool inputs as **JSON strings**.",
    "• Supported filters (NocoDB syntax): date range, ticker, transaction type.",
    "• Example: 'Show my transactions for March 2025' → {{\"filters\": [\"(Date,ge,exactDate,2025-03-01)\", \"(Date,lt,exactDate,2025-04-01)\"]}}",
    "• Example: 'Show all sells of PFC.NS' → {{\"filters\": [\"(Ticker,eq,text,PFC.NS)\", \"(TransactionType,eq,text,Sell)\"]}}",
    "• Example: 'Show all transactions today' → {{\"filters\": [\"(Date,ge,exactDate,{today_date})\", \"(Date,lt,exactDate,{tomorrow_date})\"]}}",
    "• Example: 'List all transactions' → pass an empty string '' to fetch all.",
    "• Once fetched, summarise clearly grouped by Ticker or date as relevant. Show Ticker, Type, Shares, Cost, Date, and Note if present.",

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

        self._tools = tools
        self._build_agent()
        logger.info("PortfolioManager initialized.")

    def _build_agent(self):
        def _state_modifier(state):
            now = datetime.now()
            today = now.strftime("%Y-%m-%d %A")
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            content = ("\n".join(SYSTEM_PROMPT)
                       .replace("{today_date}", today)
                       .replace("{tomorrow_date}", tomorrow))
            return [SystemMessage(content=content)] + state["messages"]

        self.agent = create_react_agent(llm, self._tools, prompt=_state_modifier, checkpointer=MemorySaver())

    def agent_portfolio(self, prompt: str) -> str:
        text = prompt.strip()
        logger.info("Received prompt: %s", text)
        for attempt in range(2):
            try:
                result = self.agent.invoke(
                    {"messages": [HumanMessage(content=text)]},
                    config={"configurable": {"thread_id": "portfolio"}},
                )
                output = result["messages"][-1].content
                logger.info("Agent response: %s", output)
                return output
            except ValueError as e:
                if "INVALID_CHAT_HISTORY" in str(e) and attempt == 0:
                    logger.warning("Corrupt portfolio chat history — resetting memory and retrying.")
                    self._build_agent()
                    continue
                logger.error("Error in agent_portfolio: %s", e, exc_info=True)
                return "An error occurred while processing your request."
            except Exception as e:
                logger.error("Error in agent_portfolio: %s", e, exc_info=True)
                return "An error occurred while processing your request."

    def add_transaction(self, data: str) -> str:
        logger.info("Adding transaction with data: %s", data)
        try:
            data_object = json.loads(data)
            ticker   = (data_object.get("Ticker") or "").strip().upper()
            tx_type  = self._normalize_transaction_type(data_object.get("TransactionType"))
            note     = str(data_object.get("Note") or "").strip()
            data_object["Ticker"] = ticker
            if tx_type:
                data_object["TransactionType"] = tx_type
            if note:
                data_object["Note"] = note
            else:
                data_object.pop("Note", None)

            shares = self._parse_number(data_object.get("NoOfShares"))
            cost = self._parse_number(data_object.get("CostPerShare"))
            if shares is not None:
                data_object["NoOfShares"] = shares
            if cost is not None:
                data_object["CostPerShare"] = cost

            response_add = self.transactions_model.create(data_object)
            logger.info("Transaction creation response: %s", response_add)
            if "failed" in str(response_add).lower():
                logger.warning("Failed to add transaction entry.")
                return "Failed to add transaction entry. Try again?"

            # Auto-create offsetting CASH entry for Buy/Sell (full proceeds, always in INR)
            if ticker != "CASH" and tx_type in ("Buy", "Sell"):
                amount    = (shares or 0.0) * (cost or 0.0)
                portfolio = (data_object.get("Portfolio") or "Default").strip()
                date_str  = data_object.get("Date", "")
                cash_type = "Withdraw" if tx_type == "Buy" else "Deposit"

                # Convert to INR if the stock trades in a foreign currency
                is_indian = ticker.upper().endswith((".NS", ".BO"))
                if not is_indian:
                    try:
                        info = yf.Ticker(ticker).info or {}
                        currency = (info.get("currency") or "INR").upper()
                        if currency != "INR":
                            fx_info = yf.Ticker(f"{currency}INR=X").info or {}
                            fx_rate = (
                                fx_info.get("regularMarketPrice")
                                or fx_info.get("currentPrice")
                                or 1.0
                            )
                            amount = round(amount * float(fx_rate), 2)
                            logger.info(
                                "Auto-cash FX conversion: %s %s × %.4f = ₹%.2f",
                                currency, shares * cost, fx_rate, amount,
                            )
                    except Exception as fx_err:
                        logger.warning("FX conversion failed for %s, storing native amount: %s", ticker, fx_err)

                cash_entry = {
                    "Ticker":          "CASH",
                    "TransactionType": cash_type,
                    "NoOfShares":      1,
                    "CostPerShare":    amount,
                    "Date":            date_str,
                    "Portfolio":       portfolio,
                }
                cash_resp = self.transactions_model.create(cash_entry)
                logger.info("Auto-cash entry (%s ₹%.0f) response: %s", cash_type, amount, cash_resp)
                if "failed" in str(cash_resp).lower():
                    logger.warning("Stock transaction saved but auto-cash entry failed.")
                    return "Transaction added successfully (⚠️ cash balance update failed — record manually)"

            return "Transaction added successfully"
        except Exception as e:
            logger.error("Error in add_transaction: %s", e, exc_info=True)
            return "Failed to add transaction due to an error."

    @staticmethod
    def _normalize_transaction_type(value) -> str:
        text = str(value or "").strip().lower()
        mapping = {
            "buy": "Buy",
            "sell": "Sell",
            "deposit": "Deposit",
            "withdraw": "Withdraw",
            "withdrawal": "Withdraw",
        }
        return mapping.get(text, str(value or "").strip())

    @staticmethod
    def _parse_number(value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned)

    def update_cmp(self) -> str:
        """Fetch latest prices from yfinance and update all transaction rows in NocoDB."""
        transactions = self.transactions_model.list()
        tickers = {
            tx.get("Ticker") for tx in transactions
            if tx.get("Ticker") and tx.get("Ticker", "").upper() != "CASH"
        }
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

    def get_dashboard_data(self) -> Dict:
        """Return a compact portfolio snapshot for Telegram dashboard views."""
        transactions = self.transactions_model.list()
        if not transactions:
            return {"as_of": datetime.now(), "portfolios": {}, "totals": {}, "holdings": [], "risk_flags": []}

        usd_to_inr = self._get_usd_to_inr()
        portfolios: Dict[str, Dict] = {}

        for tx in transactions:
            ticker = (tx.get("Ticker") or "").strip().upper()
            tx_type = (tx.get("TransactionType") or "").strip()
            shares = float(tx.get("NoOfShares") or 0)
            cost = float(tx.get("CostPerShare") or 0)
            cmp = float(tx.get("CMP") or 0)
            portfolio = (tx.get("Portfolio") or "Default").strip()

            if not ticker or shares == 0:
                continue

            entry = portfolios.setdefault(portfolio, {
                "cash": 0.0,
                "positions": {},
            })

            if ticker == "CASH":
                if tx_type == "Deposit":
                    entry["cash"] += cost
                elif tx_type in {"Withdraw", "Withdrawal"}:
                    entry["cash"] -= cost
                continue

            fx = 1.0 if ticker.endswith((".NS", ".BO")) else usd_to_inr
            pos = entry["positions"].setdefault(ticker, {
                "ticker": ticker,
                "shares_bought": 0.0,
                "shares_sold": 0.0,
                "buy_cost_inr": 0.0,
                "sell_proceeds_inr": 0.0,
                "cmp_inr": 0.0,
            })
            if cmp:
                pos["cmp_inr"] = cmp * fx
            if tx_type == "Buy":
                pos["shares_bought"] += shares
                pos["buy_cost_inr"] += shares * cost * fx
            elif tx_type == "Sell":
                pos["shares_sold"] += shares
                pos["sell_proceeds_inr"] += shares * cost * fx

        holdings: List[Dict] = []
        portfolio_rows: Dict[str, Dict] = {}
        total_invested = total_current = total_cash = 0.0

        for portfolio_name, pdata in portfolios.items():
            rows: List[Dict] = []
            invested = current = 0.0
            for ticker, pos in pdata["positions"].items():
                net_shares = pos["shares_bought"] - pos["shares_sold"]
                if net_shares <= 0:
                    continue
                avg_cost = pos["buy_cost_inr"] / pos["shares_bought"] if pos["shares_bought"] else 0.0
                invested_value = net_shares * avg_cost
                current_value = net_shares * pos["cmp_inr"]
                unrealised = current_value - invested_value
                row = {
                    "ticker": ticker,
                    "portfolio": portfolio_name,
                    "net_shares": net_shares,
                    "avg_cost_inr": avg_cost,
                    "cmp_inr": pos["cmp_inr"],
                    "invested_inr": invested_value,
                    "current_inr": current_value,
                    "unrealised_pl": unrealised,
                    "unrealised_pct": (unrealised / invested_value * 100) if invested_value else 0.0,
                }
                rows.append(row)
                holdings.append(row)
                invested += invested_value
                current += current_value

            cash = pdata["cash"]
            total_value = current + cash
            portfolio_rows[portfolio_name] = {
                "name": portfolio_name,
                "invested_inr": invested,
                "current_inr": current,
                "cash_inr": cash,
                "total_value_inr": total_value,
                "unrealised_pl": current - invested,
                "holdings": sorted(rows, key=lambda item: item["current_inr"], reverse=True),
            }
            total_invested += invested
            total_current += current
            total_cash += cash

        holdings.sort(key=lambda item: item["current_inr"], reverse=True)
        grand_total = total_current + total_cash
        for item in holdings:
            item["weight_pct"] = (item["current_inr"] / grand_total * 100) if grand_total else 0.0

        risk_flags: List[str] = []
        concentration = [h for h in holdings if h["weight_pct"] > 15]
        if concentration:
            risk_flags.append(
                "Large positions: " + ", ".join(f"{h['ticker']} {h['weight_pct']:.1f}%" for h in concentration[:3])
            )
        if total_cash < 0:
            risk_flags.append(f"Negative cash balance: ₹{total_cash:,.0f}")
        low_cmp = [h for h in holdings if h["cmp_inr"] <= 0]
        if low_cmp:
            risk_flags.append(f"{len(low_cmp)} holding(s) missing CMP updates")
        if not risk_flags:
            risk_flags.append("No immediate concentration or cash warnings.")

        return {
            "as_of": datetime.now(),
            "totals": {
                "invested_inr": total_invested,
                "current_inr": total_current,
                "cash_inr": total_cash,
                "total_value_inr": grand_total,
                "unrealised_pl": total_current - total_invested,
                "unrealised_pct": ((total_current - total_invested) / total_invested * 100) if total_invested else 0.0,
                "portfolio_count": len(portfolio_rows),
            },
            "portfolios": portfolio_rows,
            "holdings": holdings,
            "risk_flags": risk_flags,
        }

    @staticmethod
    def _get_usd_to_inr() -> float:
        usd_inr_ticker = yf.Ticker("USDINR=X")
        return (
            usd_inr_ticker.info.get("regularMarketPrice")
            or usd_inr_ticker.fast_info.get("lastPrice", 84.0)
        )

    def get_profit_loss_report(self) -> str:
        """Compute P&L across all portfolio positions and return a formatted Markdown string.

        CostPerShare and CMP from yfinance are both in the stock's native currency
        (₹ for .NS, USD for US stocks), so both need USD→INR conversion for US-listed stocks.
        """
        transactions = self.transactions_model.list()
        if not transactions:
            return "📭 No portfolio transactions found."

        usd_to_inr = self._get_usd_to_inr()
        logger.info("USD → INR rate: %s", usd_to_inr)

        ticker_data: Dict[tuple, Dict] = {}
        for tx in transactions:
            ticker    = tx.get("Ticker", "").strip()
            tx_type   = tx.get("TransactionType", "").strip()
            shares    = float(tx.get("NoOfShares") or 0)
            cost      = float(tx.get("CostPerShare") or 0)
            cmp       = float(tx.get("CMP") or 0)
            portfolio = tx.get("Portfolio", "Unknown").strip()

            if not ticker or ticker.upper() == "CASH" or shares == 0:
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

        grand_invested = grand_current = grand_real_pl = 0.0

        for portfolio_name in sorted(portfolios):
            pd = portfolios[portfolio_name]
            lines.append("")
            lines.append(f"💼 *{portfolio_name}*")

            port_invested = port_current = port_real_pl = 0.0

            inr_open   = [t for t in pd["open"]   if t["currency"] == "INR"]
            inr_closed = [t for t in pd["closed"] if t["currency"] == "INR"]
            usd_open   = [t for t in pd["open"]   if t["currency"] == "USD"]
            usd_closed = [t for t in pd["closed"] if t["currency"] == "USD"]

            if inr_open or inr_closed:
                lines.append("🇮🇳 *Indian Stocks*")
                lines.extend(build_section(inr_open, inr_closed))
                inr_inv      = sum(t["invested_inr"] for t in inr_open)
                inr_cur      = sum(t["current_inr"]  for t in inr_open)
                inr_real_pl  = sum(t["realised_pl"]  for t in inr_open) + sum(t["realised_pl"] for t in inr_closed)
                inr_unreal   = inr_cur - inr_inv
                inr_pl       = inr_unreal + inr_real_pl
                port_invested += inr_inv; port_current += inr_cur; port_real_pl += inr_real_pl
                grand_invested += inr_inv; grand_current += inr_cur; grand_real_pl += inr_real_pl
                lines.append(f"   🏦 Indian Subtotal: Deployed {fmt_inr(inr_inv)} → {fmt_inr(inr_cur)}  {pl_emoji(inr_pl)} P&L {fmt_inr(inr_pl)} (Realised: {fmt_inr(inr_real_pl)} | Unrealised: {fmt_inr(inr_unreal)})")

            if usd_open or usd_closed:
                lines.append("🇺🇸 *US Stocks (cost in ₹, CMP converted from USD)*")
                lines.extend(build_section(usd_open, usd_closed))
                usd_inv      = sum(t["invested_inr"] for t in usd_open)
                usd_cur      = sum(t["current_inr"]  for t in usd_open)
                usd_real_pl  = sum(t["realised_pl"]  for t in usd_open) + sum(t["realised_pl"] for t in usd_closed)
                usd_unreal   = usd_cur - usd_inv
                usd_pl       = usd_unreal + usd_real_pl
                port_invested += usd_inv; port_current += usd_cur; port_real_pl += usd_real_pl
                grand_invested += usd_inv; grand_current += usd_cur; grand_real_pl += usd_real_pl
                lines.append(f"   🏦 US Subtotal: Deployed {fmt_inr(usd_inv)} → {fmt_inr(usd_cur)}  {pl_emoji(usd_pl)} P&L {fmt_inr(usd_pl)} (Realised: {fmt_inr(usd_real_pl)} | Unrealised: {fmt_inr(usd_unreal)})")

            port_unreal = port_current - port_invested
            port_pl     = port_unreal + port_real_pl
            lines.append("─────────────────────────────")
            lines.append(
                f"📌 *{portfolio_name} Total*\n"
                f"   Currently Deployed: {fmt_inr(port_invested)} → {fmt_inr(port_current)}\n"
                f"   {pl_emoji(port_unreal)} Unrealised P&L: {fmt_inr(port_unreal)}"
                + (f"\n   {pl_emoji(port_real_pl)} Realised P&L:   {fmt_inr(port_real_pl)}" if port_real_pl else "")
                + f"\n   {pl_emoji(port_pl)} Total P&L:      {fmt_inr(port_pl)} "
                f"({(port_pl / (port_invested + port_real_pl) * 100) if (port_invested + port_real_pl) else 0:+.2f}%)"
            )

        grand_unreal = grand_current - grand_invested
        grand_pl     = grand_unreal + grand_real_pl
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💼 *Grand Total (INR equivalent)*\n"
            f"   Currently Deployed: {fmt_inr(grand_invested)}\n"
            f"   Current Value:      {fmt_inr(grand_current)}\n"
            f"   {pl_emoji(grand_unreal)} Unrealised P&L: {fmt_inr(grand_unreal)}\n"
            f"   {pl_emoji(grand_real_pl)} Realised P&L:   {fmt_inr(grand_real_pl)}\n"
            f"   {pl_emoji(grand_pl)} Total P&L:      {fmt_inr(grand_pl)} "
            f"({(grand_pl / (grand_invested + grand_real_pl) * 100) if (grand_invested + grand_real_pl) else 0:+.2f}%)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        return "\n".join(lines)
