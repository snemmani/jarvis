import math
from datetime import datetime
from typing import Any, Dict, List, Optional

CASH_TICKER = "CASH"


def parse_number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    cleaned = (
        str(value)
        .replace(",", "")
        .replace("₹", "")
        .replace("$", "")
        .strip()
    )
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def normalize_transaction_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "buy": "Buy",
        "bought": "Buy",
        "sell": "Sell",
        "sold": "Sell",
        "deposit": "Deposit",
        "deposited": "Deposit",
        "withdraw": "Withdraw",
        "withdrawal": "Withdraw",
        "withdrew": "Withdraw",
    }
    return mapping.get(text, str(value or "").strip())


def normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def normalize_portfolio(value: Any) -> str:
    return str(value or "Default").strip() or "Default"


def currency_for_ticker(ticker: str) -> str:
    if ticker.upper().endswith((".NS", ".BO")):
        return "INR"
    return "USD"


def fx_for_currency(currency: str, fx_rates: Optional[Dict[str, float]], default_usd_to_inr: float) -> float:
    currency = (currency or "INR").upper()
    if currency == "INR":
        return 1.0
    if fx_rates and currency in fx_rates:
        return parse_number(fx_rates[currency], default_usd_to_inr)
    return default_usd_to_inr


def _pct(value: float, base: float) -> float:
    return value / base * 100 if base else 0.0


def transaction_sort_key(tx: Dict[str, Any]) -> tuple[str, int]:
    date_str = str(tx.get("Date") or "")[:10]
    try:
        row_id = int(tx.get("Id") or 0)
    except (TypeError, ValueError):
        row_id = 0
    return date_str, row_id

def _portfolio_row(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "cash_inr": 0.0,
        "positions": {},
        "holdings": [],
        "closed_positions": [],
        "invested_inr": 0.0,
        "current_inr": 0.0,
        "realised_pl": 0.0,
        "unrealised_pl": 0.0,
        "total_pl": 0.0,
        "total_value_inr": 0.0,
        "warnings": [],
    }


def build_portfolio_ledger(
    transactions: List[Dict[str, Any]],
    fx_rates: Optional[Dict[str, float]] = None,
    default_usd_to_inr: float = 84.0,
) -> Dict[str, Any]:
    """Aggregate transactions into cash, open holdings, closed positions, and P&L.

    Uses average-cost accounting for realised P&L on sells. Stock trade cash rows
    are not inferred here; only explicit CASH ledger rows affect cash.
    """
    portfolios: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    for tx in sorted(transactions or [], key=transaction_sort_key):
        ticker = normalize_ticker(tx.get("Ticker"))
        tx_type = normalize_transaction_type(tx.get("TransactionType"))
        shares = parse_number(tx.get("NoOfShares"))
        cost_native = parse_number(tx.get("CostPerShare"))
        cmp_native = parse_number(tx.get("CMP"))
        portfolio_name = normalize_portfolio(tx.get("Portfolio"))

        if not ticker:
            continue

        portfolio = portfolios.setdefault(portfolio_name, _portfolio_row(portfolio_name))

        if ticker == CASH_TICKER:
            cash_units = shares if shares else 1.0
            amount = cash_units * cost_native
            if tx_type == "Deposit":
                portfolio["cash_inr"] += amount
            elif tx_type in {"Withdraw", "Withdrawal"}:
                portfolio["cash_inr"] -= amount
            elif amount:
                msg = f"Unknown CASH transaction type {tx_type!r} in {portfolio_name}"
                portfolio["warnings"].append(msg)
                warnings.append(msg)
            continue

        if shares <= 0:
            msg = f"Skipped {ticker} {tx_type} in {portfolio_name}: shares must be positive"
            portfolio["warnings"].append(msg)
            warnings.append(msg)
            continue

        currency = currency_for_ticker(ticker)
        fx = fx_for_currency(currency, fx_rates, default_usd_to_inr)
        key = ticker
        pos = portfolio["positions"].setdefault(key, {
            "ticker": ticker,
            "portfolio": portfolio_name,
            "currency": currency,
            "fx_rate": fx,
            "total_bought": 0.0,
            "total_sold": 0.0,
            "buy_cost_native": 0.0,
            "buy_cost_inr": 0.0,
            "open_cost_native": 0.0,
            "open_cost_inr": 0.0,
            "open_shares": 0.0,
            "realised_cost_basis_inr": 0.0,
            "sell_proceeds_native": 0.0,
            "sell_proceeds_inr": 0.0,
            "cmp_native": 0.0,
            "cmp_inr": 0.0,
            "notes": [],
            "buy_dates": [],
            "sell_dates": [],
            "all_buy_lots": [],
        })
        if cmp_native:
            pos["cmp_native"] = cmp_native
            pos["cmp_inr"] = cmp_native * fx

        date_str = str(tx.get("Date") or "")[:10]
        note = str(tx.get("Note") or "").strip()
        if note:
            pos["notes"].append({"date": date_str, "type": tx_type, "note": note})

        if tx_type == "Buy":
            pos["total_bought"] += shares
            pos["buy_cost_native"] += shares * cost_native
            pos["buy_cost_inr"] += shares * cost_native * fx
            pos["open_shares"] += shares
            pos["open_cost_native"] += shares * cost_native
            pos["open_cost_inr"] += shares * cost_native * fx
            if date_str:
                pos["buy_dates"].append(date_str)
                pos["all_buy_lots"].append((date_str, shares, cost_native))
        elif tx_type == "Sell":
            pos["total_sold"] += shares
            pos["sell_proceeds_native"] += shares * cost_native
            pos["sell_proceeds_inr"] += shares * cost_native * fx
            matched_shares = min(shares, pos["open_shares"])
            if matched_shares > 0:
                avg_open_native = pos["open_cost_native"] / pos["open_shares"]
                avg_open_inr = pos["open_cost_inr"] / pos["open_shares"]
                pos["open_shares"] -= matched_shares
                pos["open_cost_native"] -= matched_shares * avg_open_native
                pos["open_cost_inr"] -= matched_shares * avg_open_inr
                pos["realised_cost_basis_inr"] += matched_shares * avg_open_inr
                if abs(pos["open_shares"]) < 1e-9:
                    pos["open_shares"] = 0.0
                    pos["open_cost_native"] = 0.0
                    pos["open_cost_inr"] = 0.0
            if date_str:
                pos["sell_dates"].append(date_str)
        else:
            msg = f"Unknown stock transaction type {tx_type!r} for {ticker} in {portfolio_name}"
            portfolio["warnings"].append(msg)
            warnings.append(msg)

    all_holdings: List[Dict[str, Any]] = []
    all_closed: List[Dict[str, Any]] = []
    total_invested = total_current = total_cash = total_realised = 0.0

    for portfolio_name, portfolio in portfolios.items():
        holdings: List[Dict[str, Any]] = []
        closed: List[Dict[str, Any]] = []
        invested = current = realised = 0.0

        for ticker, pos in portfolio["positions"].items():
            bought = pos["total_bought"]
            sold = pos["total_sold"]
            net_shares = pos["open_shares"]
            avg_cost_native = pos["open_cost_native"] / net_shares if net_shares else 0.0
            avg_cost_inr = pos["open_cost_inr"] / net_shares if net_shares else 0.0
            cost_basis_sold_inr = pos["realised_cost_basis_inr"]
            realised_pl = pos["sell_proceeds_inr"] - cost_basis_sold_inr
            realised_pct = _pct(realised_pl, cost_basis_sold_inr)
            realised += realised_pl

            if sold > bought:
                msg = f"Oversold {ticker} in {portfolio_name}: sold {sold:g}, bought {bought:g}"
                portfolio["warnings"].append(msg)
                warnings.append(msg)

            if net_shares > 0:
                invested_value = pos["open_cost_inr"]
                current_value = net_shares * pos["cmp_inr"]
                unrealised = current_value - invested_value
                row = {
                    "ticker": ticker,
                    "portfolio": portfolio_name,
                    "currency": pos["currency"],
                    "net_shares": net_shares,
                    "total_bought": bought,
                    "total_sold": sold,
                    "avg_cost_native": avg_cost_native,
                    "avg_cost_inr": avg_cost_inr,
                    "cmp_native": pos["cmp_native"],
                    "cmp_inr": pos["cmp_inr"],
                    "invested_inr": invested_value,
                    "current_inr": current_value,
                    "unrealised_pl": unrealised,
                    "unrealised_pct": _pct(unrealised, invested_value),
                    "sell_proceeds_inr": pos["sell_proceeds_inr"],
                    "realised_pl": realised_pl,
                    "realised_pct": realised_pct,
                }
                holdings.append(row)
                all_holdings.append(row)
                invested += invested_value
                current += current_value
            elif bought > 0 or sold > 0:
                closed_basis = pos["buy_cost_inr"] if sold <= bought else cost_basis_sold_inr
                row = {
                    "ticker": ticker,
                    "portfolio": portfolio_name,
                    "currency": pos["currency"],
                    "total_bought": bought,
                    "total_sold": sold,
                    "buy_cost_inr": closed_basis,
                    "sell_inr": pos["sell_proceeds_inr"],
                    "realised_pl": realised_pl,
                    "realised_pct": realised_pct,
                }
                closed.append(row)
                all_closed.append(row)

        holdings.sort(key=lambda item: item["current_inr"], reverse=True)
        closed.sort(key=lambda item: item["realised_pl"], reverse=True)
        portfolio["holdings"] = holdings
        portfolio["closed_positions"] = closed
        portfolio["invested_inr"] = invested
        portfolio["current_inr"] = current
        portfolio["realised_pl"] = realised
        portfolio["unrealised_pl"] = current - invested
        portfolio["total_pl"] = portfolio["unrealised_pl"] + realised
        portfolio["total_value_inr"] = current + portfolio["cash_inr"]

        total_invested += invested
        total_current += current
        total_cash += portfolio["cash_inr"]
        total_realised += realised

    all_holdings.sort(key=lambda item: item["current_inr"], reverse=True)
    all_closed.sort(key=lambda item: item["realised_pl"], reverse=True)
    grand_total = total_current + total_cash
    for item in all_holdings:
        item["weight_pct"] = _pct(item["current_inr"], grand_total)

    return {
        "as_of": datetime.now(),
        "totals": {
            "invested_inr": total_invested,
            "current_inr": total_current,
            "cash_inr": total_cash,
            "realised_pl": total_realised,
            "unrealised_pl": total_current - total_invested,
            "total_pl": (total_current - total_invested) + total_realised,
            "unrealised_pct": _pct(total_current - total_invested, total_invested),
            "total_value_inr": grand_total,
            "portfolio_count": len(portfolios),
        },
        "portfolios": portfolios,
        "holdings": all_holdings,
        "closed_positions": all_closed,
        "warnings": warnings,
    }


def compute_cash_by_portfolio(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
    ledger = build_portfolio_ledger(transactions, fx_rates={"USD": 1.0}, default_usd_to_inr=1.0)
    return {name: row["cash_inr"] for name, row in ledger["portfolios"].items()}


def compute_positions_by_portfolio(transactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    raw: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tx in sorted(transactions or [], key=transaction_sort_key):
        ticker = normalize_ticker(tx.get("Ticker"))
        tx_type = normalize_transaction_type(tx.get("TransactionType"))
        shares = parse_number(tx.get("NoOfShares"))
        cost = parse_number(tx.get("CostPerShare"))
        portfolio = normalize_portfolio(tx.get("Portfolio"))
        date_str = str(tx.get("Date") or "")[:10]
        note = str(tx.get("Note") or "").strip()

        if not ticker or ticker == CASH_TICKER or shares <= 0:
            continue

        pos = raw.setdefault(portfolio, {}).setdefault(ticker, {
            "total_bought": 0.0,
            "total_sold": 0.0,
            "sell_proceeds": 0.0,
            "sell_dates": [],
            "lots": [],
            "notes": [],
        })
        if note:
            pos["notes"].append({"date": date_str, "type": tx_type, "note": note})
        if tx_type == "Buy":
            pos["total_bought"] += shares
            pos["lots"].append({"date": date_str, "shares": shares, "cost": cost})
        elif tx_type == "Sell":
            pos["total_sold"] += shares
            pos["sell_proceeds"] += shares * cost
            if date_str:
                pos["sell_dates"].append(date_str)
            remaining_to_sell = shares
            while remaining_to_sell > 1e-9 and pos["lots"]:
                lot = pos["lots"][0]
                consumed = min(remaining_to_sell, lot["shares"])
                lot["shares"] -= consumed
                remaining_to_sell -= consumed
                if lot["shares"] <= 1e-9:
                    pos["lots"].pop(0)

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for portfolio, tickers in raw.items():
        result[portfolio] = {}
        for ticker, d in tickers.items():
            open_lots = [lot for lot in d["lots"] if lot["shares"] > 1e-9]
            net = sum(lot["shares"] for lot in open_lots)
            if net <= 0:
                continue
            total_invested = sum(lot["shares"] * lot["cost"] for lot in open_lots)
            buy_dates = [lot["date"] for lot in open_lots if lot["date"]]
            all_buy_lots = [
                (lot["date"], lot["shares"], lot["cost"])
                for lot in open_lots
                if lot["date"]
            ]
            result[portfolio][ticker] = {
                "net_shares": net,
                "avg_cost": total_invested / net if net else 0.0,
                "total_invested": total_invested,
                "buy_dates": sorted(buy_dates),
                "sell_dates": sorted(d["sell_dates"]),
                "total_bought": d["total_bought"],
                "total_sold": d["total_sold"],
                "sell_proceeds": d["sell_proceeds"],
                "all_buy_lots": sorted(all_buy_lots, key=lambda x: x[0]),
                "notes": sorted(
                    d["notes"],
                    key=lambda item: ((item.get("date") or ""), (item.get("type") or "")),
                    reverse=True,
                ),
            }
    return result
