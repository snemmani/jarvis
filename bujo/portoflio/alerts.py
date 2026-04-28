import json
import logging
import re
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

import requests
import yfinance as yf

from bujo.base import OPENAI_MODEL, openai_model

logger = logging.getLogger(__name__)

_LOOKBACK_HOURS = 48        # news lookback window
_DRAWDOWN_THRESHOLD = -20.0  # % unrealised loss to trigger alert
_GAIN_THRESHOLD = 50.0       # % unrealised gain to trigger alert
_LTCG_WARN_DAYS = 14        # days before 1-year mark to warn
_DIV_WARN_DAYS = 7           # days before ex-dividend to warn
_EARNINGS_WARN_DAYS = 7     # days before earnings to warn
_52W_LOW_BUFFER_PCT = 5.0   # % above 52-week low to flag


def _compute_open_positions(transactions: List[Dict]) -> Dict[str, Dict]:
    """Aggregate transactions into open positions keyed by ticker."""
    data: Dict[str, Dict] = {}

    for tx in transactions:
        ticker = (tx.get("Ticker") or "").strip()
        tx_type = (tx.get("TransactionType") or "").strip()
        shares = float(tx.get("NoOfShares") or 0)
        cost = float(tx.get("CostPerShare") or 0)
        cmp = float(tx.get("CMP") or 0)
        date_str = (tx.get("Date") or "")[:10]

        if not ticker or shares == 0:
            continue

        if ticker not in data:
            data[ticker] = {
                "total_bought": 0.0,
                "total_sold": 0.0,
                "buy_cost": 0.0,
                "cmp": 0.0,
                "buy_dates": [],
            }

        if cmp:
            data[ticker]["cmp"] = cmp

        if tx_type == "Buy":
            data[ticker]["total_bought"] += shares
            data[ticker]["buy_cost"] += shares * cost
            if date_str:
                try:
                    data[ticker]["buy_dates"].append(
                        datetime.strptime(date_str, "%Y-%m-%d").date()
                    )
                except ValueError:
                    pass
        elif tx_type == "Sell":
            data[ticker]["total_sold"] += shares

    open_positions = {}
    for ticker, d in data.items():
        net_shares = d["total_bought"] - d["total_sold"]
        if net_shares <= 0:
            continue
        avg_cost = d["buy_cost"] / d["total_bought"] if d["total_bought"] else 0.0
        open_positions[ticker] = {
            "net_shares": net_shares,
            "avg_cost": avg_cost,
            "cmp": d["cmp"],
            "oldest_buy_date": min(d["buy_dates"]) if d["buy_dates"] else None,
        }

    return open_positions


def _check_yfinance_events(
    ticker: str,
    position: Dict,
    yf_ticker: yf.Ticker,
    info: Dict,
) -> List[str]:
    alerts = []
    today = date.today()

    # Upcoming ex-dividend
    ex_div_ts = info.get("exDividendDate")
    if ex_div_ts:
        try:
            ex_div = date.fromtimestamp(ex_div_ts)
            days_away = (ex_div - today).days
            if 0 <= days_away <= _DIV_WARN_DAYS:
                div_val = info.get("lastDividendValue") or info.get("dividendRate", "?")
                alerts.append(
                    f"💰 *Dividend*: Ex-date {ex_div} ({days_away}d away) — ₹{div_val}/share"
                )
        except Exception as e:
            logger.debug("Ex-div parse failed for %s: %s", ticker, e)

    # Upcoming earnings
    try:
        cal = yf_ticker.calendar
        if cal is not None:
            earnings_dates: List[date] = []
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    earnings_dates = ed if isinstance(ed, list) else [ed]
            else:
                for col in cal.columns:
                    try:
                        earnings_dates.append(col.date() if hasattr(col, "date") else col)
                    except Exception:
                        pass

            for ed in earnings_dates:
                if isinstance(ed, datetime):
                    ed = ed.date()
                if isinstance(ed, date):
                    days_away = (ed - today).days
                    if 0 <= days_away <= _EARNINGS_WARN_DAYS:
                        alerts.append(
                            f"📅 *Earnings*: Scheduled {ed} ({days_away}d away)"
                        )
                        break
    except Exception as e:
        logger.debug("Earnings calendar unavailable for %s: %s", ticker, e)

    # Recent splits (last 7 days)
    try:
        actions = yf_ticker.actions
        if actions is not None and not actions.empty and "Stock Splits" in actions.columns:
            cutoff = datetime.now() - timedelta(days=7)
            recent = actions[
                (actions.index >= cutoff) & (actions["Stock Splits"] > 0)
            ]
            for idx, row in recent.iterrows():
                alerts.append(f"✂️ *Stock Split*: {row['Stock Splits']}:1 on {idx.date()}")
    except Exception as e:
        logger.debug("Splits check failed for %s: %s", ticker, e)

    return alerts


def _check_derived_signals(
    ticker: str,
    position: Dict,
    info: Dict,
) -> List[str]:
    alerts = []
    avg_cost = position["avg_cost"]
    cmp = position["cmp"]
    oldest_buy = position["oldest_buy_date"]
    today = date.today()

    if avg_cost <= 0 or cmp <= 0:
        return alerts

    unrealised_pct = (cmp - avg_cost) / avg_cost * 100

    if unrealised_pct <= _DRAWDOWN_THRESHOLD:
        alerts.append(
            f"🔴 *Drawdown {unrealised_pct:+.1f}%*: CMP ₹{cmp:.2f} vs avg cost ₹{avg_cost:.2f}"
            f" — stop-loss or averaging?"
        )

    if unrealised_pct >= _GAIN_THRESHOLD:
        alerts.append(
            f"🟢 *Unrealised Gain {unrealised_pct:+.1f}%*: CMP ₹{cmp:.2f} vs avg cost ₹{avg_cost:.2f}"
            f" — consider booking partial profit"
        )

    if oldest_buy:
        one_year = oldest_buy + timedelta(days=365)
        days_to_ltcg = (one_year - today).days
        if 0 <= days_to_ltcg <= _LTCG_WARN_DAYS:
            alerts.append(
                f"🏷️ *LTCG in {days_to_ltcg}d*: Oldest lot crosses 1-year on {one_year}"
                f" — selling before this attracts STCG"
            )

    low_52w = info.get("fiftyTwoWeekLow")
    if low_52w and low_52w > 0:
        pct_above = (cmp - low_52w) / low_52w * 100
        if 0 <= pct_above <= _52W_LOW_BUFFER_PCT:
            alerts.append(
                f"📉 *Near 52W Low*: CMP ₹{cmp:.2f} is {pct_above:.1f}%"
                f" above 52W low ₹{low_52w:.2f}"
            )

    return alerts


def _fetch_news(company_name: str, ticker: str) -> List[Dict]:
    """Fetch recent news via Google News RSS."""
    clean_ticker = re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)
    query = urllib.parse.quote(f'"{company_name}" OR "{clean_ticker}" stock')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        cutoff = datetime.now() - timedelta(hours=_LOOKBACK_HOURS)
        items = []
        for item in root.findall(".//item")[:25]:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            pub_str = item.findtext("pubDate", "")
            try:
                pub_dt = parsedate_to_datetime(pub_str).replace(tzinfo=None)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass
            items.append({"title": title})
        return items
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", ticker, e)
        return []


def _classify_news(
    company_name: str, ticker: str, news_items: List[Dict]
) -> List[str]:
    """Use LLM to filter news down to HIGH-priority investor alerts."""
    if not news_items:
        return []

    headlines = "\n".join(f"- {n['title']}" for n in news_items)
    prompt = (
        f"You are a financial analyst reviewing news about {company_name} ({ticker}).\n\n"
        "Classify each headline as HIGH or LOW priority for an investor holding this stock.\n\n"
        "HIGH (always flag): auditor resignation, CFO/CEO/board member resignation or appointment, "
        "SEBI investigation or charges, promoter selling or pledging stake, debt default or credit "
        "downgrade, major litigation or court order, supply chain disruption, key customer loss, "
        "US/India/other country sanctions or trade restrictions, geopolitical events directly affecting "
        "the company, merger/acquisition announcement, bankruptcy risk, fraud allegation, "
        "regulatory ban or licence suspension, major data breach.\n\n"
        "LOW (ignore): routine earnings summaries, analyst price targets, general market commentary, "
        "technical analysis, index rebalancing.\n\n"
        f"Headlines:\n{headlines}\n\n"
        "Return ONLY a JSON array of HIGH items:\n"
        '[{"headline": "...", "reason": "one-line investor impact"}]\n'
        "If none are HIGH, return []."
    )

    try:
        resp = openai_model.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [
                f"📰 *{it['headline']}*\n   _{it.get('reason', '')}_"
                for it in items
                if it.get("headline")
            ]
    except Exception as e:
        logger.warning("News classification failed for %s: %s", ticker, e)

    return []


def run_portfolio_alerts(transactions_model) -> str:
    """
    Run all alert checks across open positions.
    Returns a formatted Telegram message, or empty string if nothing to report.
    """
    transactions = transactions_model.list()
    if not transactions:
        return ""

    positions = _compute_open_positions(transactions)
    if not positions:
        return ""

    all_alerts: Dict[str, List[str]] = {}

    for ticker, position in positions.items():
        ticker_alerts: List[str] = []

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            company_name = info.get("shortName") or ticker
        except Exception as e:
            logger.warning("yfinance fetch failed for %s: %s", ticker, e)
            yf_ticker = yf.Ticker(ticker)
            info = {}
            company_name = ticker

        ticker_alerts.extend(_check_yfinance_events(ticker, position, yf_ticker, info))
        ticker_alerts.extend(_check_derived_signals(ticker, position, info))

        news_items = _fetch_news(company_name, ticker)
        ticker_alerts.extend(_classify_news(company_name, ticker, news_items))

        if ticker_alerts:
            all_alerts[ticker] = ticker_alerts

    if not all_alerts:
        return ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🚨 *Portfolio Alerts*",
        f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for ticker, alerts in all_alerts.items():
        pos = positions[ticker]
        unrealised_pct = (
            (pos["cmp"] - pos["avg_cost"]) / pos["avg_cost"] * 100
            if pos["avg_cost"] > 0 and pos["cmp"] > 0
            else 0.0
        )
        lines.append(
            f"\n📌 *{ticker}*  {pos['net_shares']:.4g} shares"
            f" @ ₹{pos['avg_cost']:.2f} | CMP ₹{pos['cmp']:.2f}"
            f" ({unrealised_pct:+.1f}%)"
        )
        for alert in alerts:
            lines.append(f"  {alert}")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)
