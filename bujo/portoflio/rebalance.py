import json
import logging
import math
import os
import re
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yfinance as yf

from bujo.base import openai_model

logger = logging.getLogger(__name__)

_REBALANCE_MODEL         = os.environ.get("REBALANCE_OPENAI_MODEL", "gpt-5.4")
# Pricing per 1M tokens in USD — update via env vars when OpenAI changes rates
_PRICE_INPUT_PER_1M  = float(os.environ.get("REBALANCE_PRICE_INPUT_PER_1M",  "2.50"))
_PRICE_OUTPUT_PER_1M = float(os.environ.get("REBALANCE_PRICE_OUTPUT_PER_1M", "15.00"))

# ──────────────────────────────────────────────────────────────────────────────
# Self-contained system prompt — derived from Graham/GARP/Quality-Moat/Forensic
# framework. No external file dependency.
# ──────────────────────────────────────────────────────────────────────────────
_GRAHAM_SYSTEM_PROMPT = """
You are a Senior Equity Research Analyst and Portfolio Manager specialising in
long-term compounding portfolios. Apply a blended framework:
  • Benjamin Graham Value Investing
  • Growth at a Reasonable Price (GARP)
  • Quality Moat Investing
  • Forensic Accounting (Howard Schilit — Financial Shenanigans)
  • Portfolio-level capital allocation, rebalancing, and risk management

════════════════════════════════════
CORE OBJECTIVE
════════════════════════════════════
Build, evaluate, and rebalance the portfolios to maximise the probability of
achieving ≥20% long-term XIRR while remaining profitable short-term (≥10% XIRR).
Analyse at four levels:
  1. Overall combined portfolio
  2. Each individual portfolio
  3. Each stock within each portfolio
  4. Stocks duplicated across portfolios (aggregate exposure)

════════════════════════════════════
XIRR POTENTIAL CLASSIFICATION
════════════════════════════════════
Classify every holding and recommendation into:
  • Above 20% XIRR potential
  • 15–20% XIRR potential
  • 10–15% XIRR potential
  • Below 10% XIRR potential
  • Unclear / insufficient evidence

Justify using: earnings growth, revenue visibility, ROE/ROCE sustainability,
FCF growth, valuation re-rating risk, dividend contribution, balance sheet
strength, competitive advantage, management credibility, accounting quality,
sector tailwinds, and current valuation.

════════════════════════════════════
RECENT TRANSACTION AWARENESS
════════════════════════════════════
Classify every position by holding period:
  • Very Recent: 0–30 days
  • Recent: 31–90 days
  • Medium: 91–365 days
  • Long: >1 year

For Very Recent / Recent positions:
  - Default to Hold / Watchlist / Accumulate-on-Decline.
  - Recommend immediate Exit ONLY for: serious accounting red flag, governance
    crisis, major valuation mistake, broken thesis, or excessive concentration.
  - For every Trim/Exit, explicitly state: "Is this too early?" and answer:
    No — thesis broken / No — risk too high / No — position oversized /
    Yes — wait for more evidence / Yes — review after next result /
    Partial trim only.

════════════════════════════════════
ANALYTICAL PRIORITY ORDER
════════════════════════════════════
1. Probability of long-term compounding
2. Ability to contribute to 20% XIRR
3. Accounting quality and management credibility
4. Business quality and durability
5. Portfolio fit and diversification
6. Valuation discipline
7. Concentration and correlation risk
8. Growth potential

Cheapness alone is never enough. High growth alone is never enough.

════════════════════════════════════
POSITION SIZING RULES (enforce strictly)
════════════════════════════════════
  • Max single stock: 10–15% of portfolio
  • Max low-conviction stock: 3–5%
  • Min meaningful high-conviction position: 5%
  • Max cyclical allocation: 20–25%
  • Max commodity-linked: 15–20%
  • Max exposure to <15% XIRR potential stocks: 20–30%
  • Max Moderate Concern accounting quality exposure: 15–20%
  • Avoid Serious Red Flag companies unless explicitly justified as tactical/short-term

════════════════════════════════════
SECTOR-SPECIFIC VALUATION RULES
════════════════════════════════════
Banks: Use P/B, ROE, ROA, NIM, Gross NPA%, Net NPA%, Provision Coverage,
  Credit Cost, Loan Growth Quality, CASA, CAR. DO NOT use Current Ratio or
  Graham Number for banks.

NBFCs / Lending: Cost of Funds, Spread, NIM, Asset Quality, Provisioning
  conservatism, ALM risk, Borrowing concentration, Regulatory risk, CAR.
  Assess whether growth is underwriting-led or risk-led.

Technology / Software: FCF generation, Recurring revenue quality, Margin
  durability, ROCE, Buybacks, Pricing power, Customer retention, Acquisition
  discipline. Assess organic revenue growth and multiple expansion/compression.

Pharma / Specialty Chemicals / Manufacturing: ROCE, FCF conversion, Capacity
  utilisation, Regulatory risk, Export dependence, Product concentration,
  R&D productivity, Working capital. Assess operating leverage and margin expansion.

Cyclicals / Metals / Energy / Utilities: P/E, P/B, Dividend yield, FCF,
  Balance sheet, Commodity price sensitivity, Government/regulatory influence.
  Distinguish: cheap-cyclical vs structurally-impaired vs genuinely-mispriced.
  Assess re-rating potential and earnings normalisation.

════════════════════════════════════
FORENSIC ACCOUNTING SCREEN (for every holding)
════════════════════════════════════
Screen for:
  • Revenue recognition concerns
  • Weak OCF vs reported earnings (Cash Conversion Ratio <0.8 = concern; <0.6 = serious)
  • Rising receivables faster than revenue
  • Unsupported margin improvement
  • Aggressive expense capitalisation
  • Negative FCF with positive Net Income
  • Other-income dependence
  • Frequent exceptional items or adjusted earnings
  • Deteriorating cash conversion over time
  • Rising net debt or off-balance-sheet liabilities
  • Share dilution (issuance > buybacks)
  • Related-party transactions
  • Net Debt/EBITDA >3x
  • Auditor changes or qualifications
  • Promoter pledge (where data available)

Classify accounting quality: Clean / Mild Concern / Moderate Concern / Serious Red Flag.
If Moderate Concern or Serious Red Flag on a large position → default to Trim or Exit
unless very recent purchase with no thesis break.

════════════════════════════════════
FINANCIAL DELIVERY QUALITY (5-year view)
════════════════════════════════════
Note: assessment is based on 5-year financial trend data and Screener.in
qualitative signals provided — not on earnings call transcripts or management
guidance documents. Assess what the numbers and signals actually show:
  • Revenue / earnings growth consistency vs sector peers
  • Debt reduction or leverage trend
  • FCF improvement or deterioration over 5 years
  • Dividend consistency and payout discipline
  • Working capital and receivables management trend
  • Screener.in pros/cons signals (promoter holding changes, ROE trend, etc.)
Classify: Strong Delivery / Mixed Delivery / Poor Delivery.
Provide 2–3 evidence points drawn strictly from the data provided — do not
fabricate or recall management statements not present in the prompt.

════════════════════════════════════
FORWARD ASSESSMENT SOURCE PRIORITY
════════════════════════════════════
For forward-looking judgments, prioritise evidence in this order:
  1. Hard financial data and valuation metrics provided in the prompt
  2. Recent high-signal headlines and event flags provided in the prompt
  3. Stored transaction notes / thesis notes provided in the prompt
  4. Multi-year trend data and Screener.in qualitative signals

Do not invent management guidance, broker estimates, or catalysts not present in
the supplied evidence. For every forward-looking recommendation, cite at least
one concrete forward driver and one concrete forward risk from the prompt.

════════════════════════════════════
INTRINSIC VALUE CLASSIFICATION
════════════════════════════════════
Classify each stock as:
  • Discount to intrinsic value
  • Near fair value
  • Premium to intrinsic value

Use: normalised earnings, FCF, balance sheet, sector multiples, business
quality, growth durability, accounting quality, management credibility,
and capital allocation discipline.

════════════════════════════════════
RECOMMENDED STOCK CATEGORIES (for new entries)
════════════════════════════════════
Category A — Core Long-Term Compounders:
  Durable moat, high ROCE/ROE, long reinvestment runway, strong cash conversion,
  low leverage, clean accounting, credible management, reasonable valuation.

Category B — GARP Growth Opportunities:
  Growth at reasonable price, PEG <1.5 where applicable, ROE/ROCE >15%,
  strong revenue and earnings visibility, good capital allocation, clean accounting.

Category C — Opportunistic Value with Re-Rating Potential:
  Undervaluation, improving business quality, strong balance sheet, FCF
  improvement, dividend support. Should not dominate a 20% XIRR portfolio.

════════════════════════════════════
REQUIRED OUTPUT STRUCTURE (Parts A–K)
════════════════════════════════════
Part A — Executive Summary
  Overall health, top 5 risks, top 5 rebalance actions, portfolios needing urgent
  action, what is structurally wrong today.

Part B — Overall Portfolio Snapshot
  Table 1: Portfolio-Level Summary (Invested, Current, Gain%, XIRR, Fwd XIRR,
    Holdings, Largest %, Top Sector %, Health, Rebalance Urgency)
  Table 2: Top Holdings (Rank, Symbol, Current Value, % of Total, Expected XIRR, Action)
  Table 3: Sector Allocation (Sector, Value, %, Role, Suggested Target %, Risk, Action)
  Table 4: Stocks Unlikely to Support 20% XIRR (Symbol, Value, %, Expected XIRR,
    Reason, Action, Timeline)
  Table 5: Recent Transactions Requiring Patience (Symbol, Portfolio, Buy Date,
    Holding Period, Gain%, Thesis Status, Too Early to Exit?, Review Point)

Part C — Per-Portfolio Review (one section per portfolio)
  1. Portfolio Diagnosis
  2. Holdings Table (Symbol, Sector, Qty, Avg Cost, CMP, Value, Portfolio%, Gain%,
     Holding Period, Position Rank)
  3. Action Table (Symbol, Current Qty, Current%, Target%, Action, Shares to Buy/Sell,
     Trade Value, Timeline, XIRR Potential, Conviction, Justification)
     Action: HOLD / TRIM / EXIT / ENTER / ACCUMULATE / WATCHLIST
     Timeline: Immediate | 1–2 weeks | 1 month | 3 months | 6 months |
               On decline only | After next quarterly result | After annual report review
  4. Recent Transaction Review (per position bought ≤90 days ago)
  5. Portfolio Rebalance Plan (phased: what to sell first → what to buy → what to wait)
  6. Target Portfolio After Rebalance (Symbol, Target Qty, Target Value, Target%, Role,
     XIRR Potential)
     Role: Core compounder | High-growth compounder | Quality financial |
           Value anchor | Cyclical opportunity | Dividend income |
           Tactical position | Turnaround | Watchlist only | Exit candidate
  7. Portfolio-Level Conclusion

Part D — Recommended New Stocks (Category A / B / C with write-ups)

Part E — 5-Year Financial Delivery & Forensic Review (every current holding)
  Financial delivery quality (from trend data + Screener.in signals),
  Shenanigans screen, accounting quality classification.
  Do NOT fabricate management statements — use only data provided.

Part F — Sector Valuation Applied (confirm which framework used per stock)

Part G — Intrinsic Value Classification (Discount / Near / Premium for each)

Part H — Rebalance Framework
  Step 1: Target allocation ranges by category
  Step 2: Position sizing violations
  Step 3: Full trade table (Current % → Target % → Shares → Timeline)
  Step 4: Phased plan:
    Phase 1 Immediate (2 weeks): Risk reduction exits/trims
    Phase 2 (1–3 months): Quality and growth upgrades
    Phase 3 (3–6 months): Opportunistic accumulation
    Phase 4 (After next results): Review watchlisted / recent positions

Part I — Final Action Table (all portfolios)
  Columns: Portfolio | Action Type | Symbol | Buy/Sell/Hold | Shares |
           Trade Value | Timeline | Priority (Urgent/High/Medium/Low) | XIRR Impact | Reason
  Action Type: Risk reduction | Quality upgrade | Growth compounding |
               Valuation discipline | Concentration reduction | Sector diversification |
               Capital deployment | Watchlist review

Part J — Target Portfolio View (post-rebalance state for each portfolio)
  Columns: Portfolio | Symbol | Target Qty | Target % | Role | Conviction |
           XIRR Potential | Review Trigger

Part K — Final Conclusion
  Stocks to Buy | Accumulate on Dips | Trim | Exit | Hold |
  Recent Purchases to Review Later | Cash Deployment Plan

════════════════════════════════════
OUTPUT QUALITY RULES
════════════════════════════════════
1. Be direct and action-oriented — no vague "Hold" without justification.
2. Every Trim/Exit/Buy/Accumulate must include exact share count.
3. Every action must include a specific timeline.
4. Every large position (>8% of portfolio) must be evaluated more deeply.
5. Every recent transaction (≤90 days) must be reviewed before recommending exit.
6. State explicitly when trim/exit is too early.
7. Separate: valuation risk, business risk, accounting risk, governance risk,
   portfolio risk, and XIRR risk.
8. Do not recommend a stock only because it is cheap.
9. Do not recommend buying a high-quality company at any price.
10. Where data is uncertain, say so clearly — do not fabricate.
11. Format the entire response as clean Markdown suitable for saving as a .md file.
12. Use tables wherever the output structure calls for them.
"""

_CASH_TICKER = "CASH"

# GrahamPrompt position-size rules (Part H, Step 2)
_MAX_SINGLE_STOCK_PCT   = 15.0
_MAX_LOW_CONVICTION_PCT = 5.0
_MAX_CYCLICAL_PCT       = 25.0
_MAX_BELOW_15_XIRR_PCT  = 30.0   # max portfolio exposure to <15% XIRR potential stocks
_MIN_MEANINGFUL_PCT     = 5.0    # below this a "meaningful" position is undersized

# Holding period buckets (GrahamPrompt – Recent Transaction Awareness)
_HP_VERY_RECENT = 30
_HP_RECENT      = 90
_HP_MEDIUM      = 365
_CANDIDATE_NEWS_LOOKBACK_HOURS = 72
_FORWARD_RISK_CONS_PHRASES = (
    "profit declined",
    "profits declined",
    "earnings declined",
    "margin declined",
    "margins declined",
    "regulatory",
    "warning letter",
    "usfda",
    "price erosion",
    "customer concentration",
    "product concentration",
    "working capital",
    "debtor days",
    "inventory days",
)
_NEWS_RISK_KEYWORDS = (
    "investigation",
    "fraud",
    "resignation",
    "warning letter",
    "usfda",
    "import alert",
    "pledge",
    "downgrade",
    "default",
    "sanction",
    "ban",
    "litigation",
    "search and seizure",
)
_FORWARD_POSITIVE_HEADLINE_KEYWORDS = (
    "order win",
    "order book",
    "contract",
    "approval",
    "launch",
    "expansion",
    "capacity",
    "partnership",
    "acquisition",
    "beat",
    "guidance raised",
)
_FORWARD_NEGATIVE_HEADLINE_KEYWORDS = (
    "investigation",
    "fraud",
    "resignation",
    "downgrade",
    "default",
    "warning letter",
    "ban",
    "sanction",
    "litigation",
    "margin pressure",
    "profit warning",
    "pledge",
)
_FORWARD_NOTE_RISK_KEYWORDS = (
    "exit if",
    "trim if",
    "re-evaluate if",
    "governance",
    "warning",
    "risk",
    "monitor",
    "npa",
    "margin",
    "credit cost",
    "rbi",
    "sebi",
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _pct(val) -> Optional[float]:
    if val is None:
        return None
    return round(float(val) * 100, 2)


def _safe_div(a, b) -> Optional[float]:
    try:
        if b and b != 0:
            return round(a / b, 4)
    except Exception:
        pass
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_percent_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if abs(num) <= 1:
        num *= 100
    return round(num, 2)


def _cr_value(raw: Any, fx_rate: float) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return round(float(raw) * fx_rate / 1e7, 2)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    if value is None:
        return "Insufficient evidence"
    try:
        return f"{prefix}{float(value):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "Insufficient evidence"


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    return _fmt_num(value, decimals=decimals, suffix="%")


def _fmt_ratio(value: Any, decimals: int = 2, suffix: str = "x") -> str:
    return _fmt_num(value, decimals=decimals, suffix=suffix)


def _fmt_text(value: Any) -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return text or "Insufficient evidence"


def _classify_holding_period(days: int) -> str:
    if days <= _HP_VERY_RECENT:
        return "Very Recent (0–30 days)"
    if days <= _HP_RECENT:
        return "Recent (31–90 days)"
    if days <= _HP_MEDIUM:
        return "Medium (91–365 days)"
    return "Long (>1 year)"


def _ltcg_status(oldest_buy: str, today: date) -> str:
    """Returns LTCG eligibility status and days to 1-year mark."""
    try:
        buy_date = date.fromisoformat(oldest_buy)
        ltcg_date = buy_date + timedelta(days=365)
        days_left = (ltcg_date - today).days
        if days_left <= 0:
            return "LTCG eligible"
        return f"STCG — LTCG in {days_left} days (on {ltcg_date})"
    except Exception:
        return "Unknown"


def _cagr(start_val: float, end_val: float, years: float) -> Optional[float]:
    try:
        if start_val > 0 and end_val > 0 and years > 0:
            return round(((end_val / start_val) ** (1 / years) - 1) * 100, 2)
    except Exception:
        pass
    return None


# Module-level FX rate cache (refreshed per process run)
_FX_CACHE: Dict[str, float] = {}


def _fetch_fx_rate(from_currency: str) -> float:
    """Fetch current exchange rate to INR, cached per process."""
    key = from_currency
    if key in _FX_CACHE:
        return _FX_CACHE[key]
    rate = 1.0
    try:
        ticker_sym = f"{from_currency}INR=X"
        yf_obj = yf.Ticker(ticker_sym)
        try:
            rate = float(yf_obj.fast_info.last_price or 0) or None
        except Exception:
            rate = None
        if not rate:
            info = yf_obj.info or {}
            rate = info.get("regularMarketPrice") or info.get("currentPrice")
        rate = float(rate) if rate else 1.0
    except Exception:
        rate = 1.0
    logger.info("FX rate %sINR = %.4f", from_currency, rate)
    _FX_CACHE[key] = rate
    return rate


def _parse_analyst_rating(raw_rating: Any) -> Optional[float]:
    if raw_rating in (None, ""):
        return None
    try:
        return float(raw_rating)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(\d+(?:\.\d+)?)", str(raw_rating))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _classify_sector(sector: str, industry: str) -> str:
    """Map yfinance sector to GrahamPrompt valuation framework bucket."""
    s = (sector or "").lower()
    i = (industry or "").lower()
    if "bank" in s or "bank" in i:
        return "Banks"
    if any(x in s for x in ["financial", "insurance", "capital market"]):
        return "Financials / NBFC"
    if any(x in s for x in ["technology", "software", "semiconductor"]):
        return "Technology"
    if any(x in s for x in ["health", "pharma", "biotech"]):
        return "Pharma / Healthcare"
    if any(x in s for x in ["energy", "oil", "gas", "utilities"]):
        return "Energy / Utilities"
    if any(x in s for x in ["material", "metal", "mining", "chemical"]):
        return "Metals / Chemicals"
    if any(x in s for x in ["industrial", "manufactur", "engineer"]):
        return "Manufacturing / Industrials"
    if any(x in s for x in ["consumer", "retail", "food", "beverage"]):
        return "Consumer"
    if any(x in s for x in ["real estate", "reit"]):
        return "Real Estate"
    return sector or "Unknown"


def _revenue_cagr_from_financials(years_data: List[Dict]) -> Optional[float]:
    """Compute revenue CAGR from oldest-to-newest pair in annual_financials list."""
    valid = [y for y in years_data if y.get("total_revenue")]
    if len(valid) < 2:
        return None
    return _cagr(valid[-1]["total_revenue"], valid[0]["total_revenue"], len(valid) - 1)


def _net_income_cagr_from_financials(years_data: List[Dict]) -> Optional[float]:
    valid = [y for y in years_data if y.get("net_income")]
    if len(valid) < 2:
        return None
    return _cagr(valid[-1]["net_income"], valid[0]["net_income"], len(valid) - 1)


def _cash_conversion_ratio(fin_data: List[Dict], cf_data: List[Dict]) -> Optional[float]:
    """OCF / Net Income (latest year). ≥0.8 = clean; <0.6 = serious concern."""
    if not fin_data or not cf_data:
        return None
    ni  = fin_data[0].get("net_income")
    ocf = cf_data[0].get("operating_cash_flow")
    if ni and ni > 0 and ocf is not None:
        return round(ocf / ni, 2)
    return None


# ──────────────────────────────────────────────
# Transaction → Position aggregation
# ──────────────────────────────────────────────

def _compute_positions_by_portfolio(
    transactions: List[Dict],
) -> Dict[str, Dict[str, Dict]]:
    """
    Returns:
      {portfolio: {ticker: {net_shares, avg_cost, total_invested,
                            buy_dates, sell_dates, total_bought, total_sold,
                            sell_proceeds, all_buy_lots, notes}}}
    all_buy_lots: list of (date_str, shares, cost) for XIRR approximation.
    """
    raw: Dict[str, Dict[str, Dict]] = {}

    for tx in transactions:
        ticker    = (tx.get("Ticker") or "").strip()
        tx_type   = (tx.get("TransactionType") or "").strip()
        shares    = float(tx.get("NoOfShares") or 0)
        cost      = float(tx.get("CostPerShare") or 0)
        portfolio = (tx.get("Portfolio") or "Default").strip()
        date_str  = (tx.get("Date") or "")[:10]
        note      = (tx.get("Note") or "").strip()

        if not ticker or ticker.upper() == _CASH_TICKER or shares == 0:
            continue

        raw.setdefault(portfolio, {})
        raw[portfolio].setdefault(ticker, {
            "total_bought":  0.0,
            "total_sold":    0.0,
            "buy_cost":      0.0,
            "sell_proceeds": 0.0,
            "buy_dates":     [],
            "sell_dates":    [],
            "all_buy_lots":  [],
            "notes":         [],
        })

        pos = raw[portfolio][ticker]
        if note:
            pos["notes"].append({
                "date": date_str,
                "type": tx_type,
                "note": note,
            })
        if tx_type == "Buy":
            pos["total_bought"] += shares
            pos["buy_cost"]     += shares * cost
            pos["buy_dates"].append(date_str)
            pos["all_buy_lots"].append((date_str, shares, cost))
        elif tx_type == "Sell":
            pos["total_sold"]      += shares
            pos["sell_proceeds"]   += shares * cost
            pos["sell_dates"].append(date_str)

    result: Dict[str, Dict[str, Dict]] = {}
    for portfolio, tickers in raw.items():
        result[portfolio] = {}
        for ticker, d in tickers.items():
            net = d["total_bought"] - d["total_sold"]
            if net <= 0:
                continue
            avg_cost = d["buy_cost"] / d["total_bought"] if d["total_bought"] else 0.0
            result[portfolio][ticker] = {
                "net_shares":    net,
                "avg_cost":      avg_cost,
                "total_invested": net * avg_cost,
                "buy_dates":     sorted(d["buy_dates"]),
                "sell_dates":    sorted(d["sell_dates"]),
                "total_bought":  d["total_bought"],
                "total_sold":    d["total_sold"],
                "sell_proceeds": d["sell_proceeds"],
                "all_buy_lots":  sorted(d["all_buy_lots"], key=lambda x: x[0]),
                "notes":         sorted(
                    d["notes"],
                    key=lambda item: ((item.get("date") or ""), (item.get("type") or "")),
                    reverse=True,
                ),
            }

    return result


def _compute_cash_by_portfolio(transactions: List[Dict]) -> Dict[str, float]:
    """
    Returns {portfolio: net_cash_balance} from explicit CASH ticker rows.
    TransactionType 'Deposit' adds cash; 'Withdraw' subtracts it.
    Amount = NoOfShares * CostPerShare (bot always records NoOfShares=1).
    """
    balances: Dict[str, float] = {}
    for tx in transactions:
        ticker = (tx.get("Ticker") or "").strip().upper()
        if ticker != _CASH_TICKER:
            continue
        tx_type   = (tx.get("TransactionType") or "").strip()
        amount    = float(tx.get("NoOfShares") or 0) * float(tx.get("CostPerShare") or 0)
        portfolio = (tx.get("Portfolio") or "Default").strip()
        balances.setdefault(portfolio, 0.0)
        if tx_type == "Deposit":
            balances[portfolio] += amount
        elif tx_type in {"Withdraw", "Withdrawal"}:
            balances[portfolio] -= amount
    return balances


# ──────────────────────────────────────────────
# Market data fetch
# ──────────────────────────────────────────────

def _fetch_ticker_data(ticker: str) -> Dict:
    """
    Fetch fundamental + technical + forensic-accounting data from yfinance.
    Returns a rich dict aligned to GrahamPrompt.md criteria across all Parts.
    """
    try:
        yf_obj = yf.Ticker(ticker)
        info   = yf_obj.info or {}

        sector   = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        sector_bucket = _classify_sector(sector, industry)

        # ── Currency detection & FX conversion ──
        currency = (info.get("currency") or "INR").upper()
        fx_rate  = _fetch_fx_rate(currency) if currency != "INR" else 1.0

        cmp_native = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # ── Core identity & price ──
        d: Dict = {
            "company_name":    info.get("shortName") or info.get("longName") or ticker,
            "sector":          sector,
            "industry":        industry,
            "sector_bucket":   sector_bucket,
            "exchange":        info.get("exchange", "N/A"),
            "currency":        currency,
            "fx_rate_to_inr":  fx_rate,
            "cmp_native":      round(cmp_native, 4) if cmp_native else None,
            "market_cap_cr":   _cr_value(info.get("marketCap"), fx_rate),
            "cmp":             round(cmp_native * fx_rate, 4) if cmp_native else None,
            "52w_high":        round(float(info.get("fiftyTwoWeekHigh")) * fx_rate, 4) if info.get("fiftyTwoWeekHigh") not in (None, "") else None,
            "52w_low":         round(float(info.get("fiftyTwoWeekLow")) * fx_rate, 4) if info.get("fiftyTwoWeekLow") not in (None, "") else None,
            "50d_ma":          round(float(info.get("fiftyDayAverage")) * fx_rate, 4) if info.get("fiftyDayAverage") not in (None, "") else None,
            "200d_ma":         round(float(info.get("twoHundredDayAverage")) * fx_rate, 4) if info.get("twoHundredDayAverage") not in (None, "") else None,
            "beta":            info.get("beta"),
            "avg_volume":      info.get("averageVolume"),
        }

        # ── Valuation multiples (Part F + Part G) ──
        d.update({
            "pe_ttm":        info.get("trailingPE"),
            "pe_forward":    info.get("forwardPE"),
            "pb":            info.get("priceToBook"),
            "ps":            info.get("priceToSalesTrailing12Months"),
            "ev_ebitda":     info.get("enterpriseToEbitda"),
            "ev_revenue":    info.get("enterpriseToRevenue"),
            "peg":           info.get("pegRatio"),
            "book_value":    round(float(info.get("bookValue")) * fx_rate, 4) if info.get("bookValue") not in (None, "") else None,
            "eps_ttm":       round(float(info.get("trailingEps")) * fx_rate, 4) if info.get("trailingEps") not in (None, "") else None,
            "eps_forward":   round(float(info.get("forwardEps")) * fx_rate, 4) if info.get("forwardEps") not in (None, "") else None,
        })

        # ── Profitability & quality (Part E + Part F) ──
        d.update({
            "roe":               _pct(info.get("returnOnEquity")),
            "roa":               _pct(info.get("returnOnAssets")),
            "profit_margin":     _pct(info.get("profitMargins")),
            "operating_margin":  _pct(info.get("operatingMargins")),
            "gross_margin":      _pct(info.get("grossMargins")),
            "ebitda_margin":     _pct(_safe_div(info.get("ebitda"), info.get("totalRevenue"))),
        })

        # ── Growth (Part D – GARP criteria) ──
        d.update({
            "revenue_growth_yoy":  _pct(info.get("revenueGrowth")),
            "earnings_growth_yoy": _pct(info.get("earningsGrowth")),
        })

        # ── Dividend (Part A, value anchor / income role) ──
        d.update({
            "dividend_yield":  _normalize_percent_value(info.get("dividendYield")),
            "dividend_rate":   round(float(info.get("dividendRate")) * fx_rate, 4) if info.get("dividendRate") not in (None, "") else None,
            "payout_ratio":    _normalize_percent_value(info.get("payoutRatio")),
        })

        # ── Balance sheet strength (Part E – Shenanigans + Part F – sector rules) ──
        total_debt = info.get("totalDebt") or 0
        total_cash = info.get("totalCash") or 0
        market_cap = info.get("marketCap") or 0
        total_revenue = info.get("totalRevenue") or 0
        d.update({
            "debt_to_equity":    info.get("debtToEquity"),
            "current_ratio":     info.get("currentRatio"),
            "quick_ratio":       info.get("quickRatio"),
            "total_debt_cr":     _cr_value(total_debt, fx_rate),
            "cash_cr":           _cr_value(total_cash, fx_rate),
            "net_debt_cr":       _cr_value((total_debt - total_cash), fx_rate) if total_debt or total_cash else None,
            "total_revenue_cr":  _cr_value(total_revenue, fx_rate),
            "net_debt_to_ebitda": _safe_div(
                total_debt - total_cash,
                info.get("ebitda") or 0,
            ),
        })

        # ── Cash flow quality (Part E – Shenanigans: OCF vs Net Income) ──
        ocf = info.get("operatingCashflow") or 0
        fcf = info.get("freeCashflow") or 0
        d.update({
            "operating_cashflow_cr": _cr_value(ocf, fx_rate),
            "free_cash_flow_cr":     _cr_value(fcf, fx_rate),
            "fcf_yield_pct":         _pct(_safe_div(fcf, market_cap)),
            "capex_cr":              _cr_value((ocf - fcf), fx_rate) if ocf not in (None, 0) and fcf not in (None, 0) else None,
        })

        # ── Ownership & governance (Part E) ──
        d.update({
            "insider_ownership_pct":       _normalize_percent_value(info.get("heldPercentInsiders")),
            "institutional_ownership_pct": _normalize_percent_value(info.get("heldPercentInstitutions")),
            "shares_outstanding_cr":       round(float(info.get("sharesOutstanding")) / 1e7, 4) if info.get("sharesOutstanding") not in (None, "") else None,
            "float_shares_cr":             round(float(info.get("floatShares")) / 1e7, 4) if info.get("floatShares") not in (None, "") else None,
            "short_ratio":                 info.get("shortRatio"),
            "shares_short_pct_float":      _normalize_percent_value(info.get("shortPercentOfFloat")),
        })

        # ── Sector-specific extras (Part F) ──
        d["net_interest_margin"] = None  # not directly available in yfinance
        # Interest coverage computed after financials are fetched (see below)

        # ── 5-year price CAGR from monthly history ──
        try:
            hist = yf_obj.history(period="5y", interval="1mo")
            if not hist.empty:
                s = hist["Close"].iloc[0]
                e = hist["Close"].iloc[-1]
                yrs = len(hist) / 12
                d["price_cagr_5y_pct"] = _cagr(s, e, yrs)
                # 52W price change
                recent = yf_obj.history(period="1y", interval="1mo")
                if not recent.empty:
                    d["price_return_1y_pct"] = _cagr(
                        recent["Close"].iloc[0], recent["Close"].iloc[-1], 1
                    )
                else:
                    d["price_return_1y_pct"] = None
            else:
                d["price_cagr_5y_pct"] = None
                d["price_return_1y_pct"] = None
        except Exception:
            d["price_cagr_5y_pct"] = None
            d["price_return_1y_pct"] = None

        # ── Annual financials: revenue, net income, EPS (last 4–5 years) ──
        annual_fin: List[Dict] = []
        annual_bs:  List[Dict] = []
        annual_cf:  List[Dict] = []
        try:
            fin = yf_obj.financials
            if fin is not None and not fin.empty:
                rows = {r: fin.loc[r] for r in fin.index if r in {
                    "Total Revenue", "Net Income", "EBIT", "EBITDA",
                    "Gross Profit", "Interest Expense Non Operating",
                }}
                for col in fin.columns[:5]:
                    yr = str(col)[:10]
                    entry = {"year": yr}
                    for label, row in rows.items():
                        try:
                            v = float(row[col])
                            entry[label.lower().replace(" ", "_")] = round(v * fx_rate / 1e7, 2)
                        except Exception:
                            entry[label.lower().replace(" ", "_")] = None
                    annual_fin.append(entry)
        except Exception as exc:
            logger.debug("Financials unavailable for %s: %s", ticker, exc)
        d["annual_financials"] = annual_fin

        # Interest coverage from financials (EBIT / Interest Expense Non Operating)
        if annual_fin:
            ebit_val = annual_fin[0].get("ebit") or 0
            # stored key for "Interest Expense Non Operating"
            int_exp_val = annual_fin[0].get("interest_expense_non_operating") or 0
            d["interest_coverage"] = (
                round(ebit_val / abs(int_exp_val), 2)
                if int_exp_val and int_exp_val != 0 else None
            )
        else:
            d["interest_coverage"] = None

        # ── Annual balance sheet: receivables, inventory, equity, debt ──
        try:
            bs = yf_obj.balance_sheet
            if bs is not None and not bs.empty:
                bs_rows = {r: bs.loc[r] for r in bs.index if r in {
                    "Accounts Receivable", "Inventory", "Total Assets",
                    "Stockholders Equity", "Long Term Debt", "Current Debt",
                }}
                for col in bs.columns[:5]:
                    yr = str(col)[:10]
                    entry = {"year": yr}
                    for label, row in bs_rows.items():
                        try:
                            v = float(row[col])
                            entry[label.lower().replace(" ", "_")] = round(v * fx_rate / 1e7, 2)
                        except Exception:
                            entry[label.lower().replace(" ", "_")] = None
                    annual_bs.append(entry)
        except Exception as exc:
            logger.debug("Balance sheet unavailable for %s: %s", ticker, exc)
        d["annual_balance_sheet"] = annual_bs

        # ── Annual cash flows: OCF, FCF, capex ──
        try:
            cf = yf_obj.cashflow
            if cf is not None and not cf.empty:
                cf_rows = {r: cf.loc[r] for r in cf.index if r in {
                    "Operating Cash Flow", "Free Cash Flow",
                    "Capital Expenditure", "Net Issuance Payments Of Debt",
                    "Cash Dividends Paid",
                }}
                for col in cf.columns[:5]:
                    yr = str(col)[:10]
                    entry = {"year": yr}
                    for label, row in cf_rows.items():
                        try:
                            v = float(row[col])
                            entry[label.lower().replace(" ", "_")] = round(v * fx_rate / 1e7, 2)
                        except Exception:
                            entry[label.lower().replace(" ", "_")] = None
                    annual_cf.append(entry)
        except Exception as exc:
            logger.debug("Cash flow unavailable for %s: %s", ticker, exc)
        d["annual_cashflows"] = annual_cf

        # ── Derived multi-year metrics ──
        d["revenue_cagr_3y_pct"]    = _revenue_cagr_from_financials(annual_fin[:3] if len(annual_fin) >= 3 else annual_fin)
        d["revenue_cagr_5y_pct"]    = _revenue_cagr_from_financials(annual_fin)
        d["net_income_cagr_3y_pct"] = _net_income_cagr_from_financials(annual_fin[:3] if len(annual_fin) >= 3 else annual_fin)
        d["net_income_cagr_5y_pct"] = _net_income_cagr_from_financials(annual_fin)
        d["cash_conversion_ratio"]  = _cash_conversion_ratio(annual_fin, annual_cf)

        # FCF/Revenue margin (latest year)
        if annual_fin and annual_cf:
            rev_latest = annual_fin[0].get("total_revenue")
            fcf_latest = annual_cf[0].get("free_cash_flow")
            d["fcf_margin_pct"] = _pct(_safe_div(fcf_latest, rev_latest)) if (rev_latest and fcf_latest) else None
        else:
            d["fcf_margin_pct"] = None

        # ── Forensic accounting signals (Part E – Shenanigans) ──
        d["forensic_flags"] = _compute_forensic_flags(d, annual_fin, annual_bs, annual_cf)

        # Business description (truncated)
        d["business_summary"] = (info.get("longBusinessSummary") or "")[:600]

        # Screener.in qualitative signals (pros/cons/about)
        d["screener_data"] = _fetch_screener_data(ticker)

        d["valuation"] = _compute_valuation_profile(d)
        return d

    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return {"company_name": ticker, "cmp": 0, "error": str(e)}


def _select_valuation_method(td: Dict[str, Any]) -> str:
    bucket = td.get("sector_bucket") or ""
    if bucket in {"Banks", "Financials / NBFC"}:
        return "pb_roe"
    if bucket in {"Energy / Utilities", "Metals / Chemicals", "Real Estate"}:
        return "normalized_pe"
    return "dcf"


def _estimate_cost_of_equity(td: Dict[str, Any]) -> float:
    beta = td.get("beta")
    beta = float(beta) if beta not in (None, "") else 1.0
    beta = _clamp(beta, 0.6, 1.8)
    return round(0.07 + beta * 0.06, 4)


def _estimate_terminal_growth(td: Dict[str, Any]) -> float:
    revenue_cagr = td.get("revenue_cagr_5y_pct")
    if revenue_cagr is None:
        revenue_cagr = td.get("revenue_cagr_3y_pct")
    if revenue_cagr is None:
        return 0.04
    return round(_clamp(float(revenue_cagr) / 100 * 0.25, 0.03, 0.055), 4)


def _estimate_initial_growth(td: Dict[str, Any]) -> float:
    candidates = [
        td.get("revenue_growth_yoy"),
        td.get("earnings_growth_yoy"),
        td.get("revenue_cagr_3y_pct"),
        td.get("net_income_cagr_3y_pct"),
    ]
    usable = [float(v) for v in candidates if v is not None]
    if not usable:
        return 0.10
    avg = sum(usable) / len(usable)
    return round(_clamp(avg / 100, 0.04, 0.18), 4)


def _compute_dcf_value(td: Dict[str, Any]) -> Dict[str, Any]:
    fcf_cr = td.get("free_cash_flow_cr")
    shares_cr = td.get("shares_outstanding_cr")
    cmp = td.get("cmp") or 0
    if not fcf_cr or fcf_cr <= 0 or not shares_cr or shares_cr <= 0:
        return {"applicable": False, "reason": "Insufficient positive FCF/share data for DCF"}

    cost = _estimate_cost_of_equity(td)
    terminal = _estimate_terminal_growth(td)
    if cost <= terminal:
        cost = terminal + 0.03
    initial = _estimate_initial_growth(td)
    scenarios = {
        "bear": (_clamp(initial - 0.04, 0.02, 0.14), _clamp(terminal - 0.005, 0.025, 0.05)),
        "base": (initial, terminal),
        "bull": (_clamp(initial + 0.03, 0.05, 0.22), _clamp(terminal + 0.005, 0.03, 0.06)),
    }
    per_share: Dict[str, float] = {}
    for name, (growth, term_growth) in scenarios.items():
        pv = 0.0
        cash_flow = float(fcf_cr)
        for year in range(1, 6):
            fade_growth = growth + (term_growth - growth) * ((year - 1) / 4)
            cash_flow *= (1 + fade_growth)
            pv += cash_flow / ((1 + cost) ** year)
        terminal_value = (cash_flow * (1 + term_growth)) / max(cost - term_growth, 0.01)
        pv += terminal_value / ((1 + cost) ** 5)
        per_share[name] = round(pv / shares_cr, 2)

    base_value = per_share["base"]
    mos = round((base_value / cmp - 1) * 100, 1) if cmp else None
    return {
        "applicable": True,
        "method": "DCF",
        "cost_of_equity_pct": round(cost * 100, 2),
        "terminal_growth_pct": round(terminal * 100, 2),
        "initial_growth_pct": round(initial * 100, 2),
        "per_share": per_share,
        "margin_of_safety_pct": mos,
    }


def _compute_reverse_dcf(td: Dict[str, Any]) -> Dict[str, Any]:
    fcf_cr = td.get("free_cash_flow_cr")
    market_cap_cr = td.get("market_cap_cr")
    if not fcf_cr or fcf_cr <= 0 or not market_cap_cr or market_cap_cr <= 0:
        return {"applicable": False, "reason": "Insufficient positive FCF/market-cap data for reverse DCF"}

    cost = _estimate_cost_of_equity(td)
    terminal = _estimate_terminal_growth(td)
    if cost <= terminal:
        cost = terminal + 0.03

    def implied_value(growth: float) -> float:
        pv = 0.0
        cash_flow = float(fcf_cr)
        for year in range(1, 6):
            cash_flow *= (1 + growth)
            pv += cash_flow / ((1 + cost) ** year)
        terminal_value = (cash_flow * (1 + terminal)) / max(cost - terminal, 0.01)
        pv += terminal_value / ((1 + cost) ** 5)
        return pv

    low, high = -0.10, 0.40
    for _ in range(50):
        mid = (low + high) / 2
        if implied_value(mid) < market_cap_cr:
            low = mid
        else:
            high = mid
    implied_growth = round(((low + high) / 2) * 100, 2)
    return {
        "applicable": True,
        "method": "Reverse DCF",
        "cost_of_equity_pct": round(cost * 100, 2),
        "terminal_growth_pct": round(terminal * 100, 2),
        "implied_fcf_growth_5y_pct": implied_growth,
    }


def _compute_pb_roe_value(td: Dict[str, Any]) -> Dict[str, Any]:
    book_value = td.get("book_value")
    roe = td.get("roe")
    pb = td.get("pb")
    cmp = td.get("cmp") or 0
    if not book_value or book_value <= 0 or roe is None:
        return {"applicable": False, "reason": "Missing book value / ROE for P/B-ROE valuation"}

    cost = _estimate_cost_of_equity(td)
    roe_dec = float(roe) / 100
    payout = td.get("payout_ratio")
    payout_dec = _clamp((float(payout) / 100) if payout is not None else 0.20, 0.05, 0.60)
    growth = _clamp(roe_dec * (1 - payout_dec) * 0.55, 0.04, 0.12)
    justified_pb = max((roe_dec - growth) / max(cost - growth, 0.01), 0.4)
    fair_value = round(book_value * justified_pb, 2)
    implied_roe = None
    if pb:
        implied_roe = round((float(pb) * (cost - growth) + growth) * 100, 2)
    mos = round((fair_value / cmp - 1) * 100, 1) if cmp else None
    return {
        "applicable": True,
        "method": "P/B-ROE",
        "cost_of_equity_pct": round(cost * 100, 2),
        "growth_pct": round(growth * 100, 2),
        "justified_pb": round(justified_pb, 2),
        "fair_value_per_share": fair_value,
        "margin_of_safety_pct": mos,
        "implied_sustainable_roe_pct": implied_roe,
    }


def _compute_normalized_pe_value(td: Dict[str, Any]) -> Dict[str, Any]:
    eps_values = [v for v in [td.get("eps_ttm"), td.get("eps_forward")] if v not in (None, 0)]
    cmp = td.get("cmp") or 0
    if not eps_values:
        return {"applicable": False, "reason": "Missing EPS data for normalized P/E valuation"}

    normalized_eps = sum(float(v) for v in eps_values) / len(eps_values)
    growth = td.get("earnings_growth_yoy")
    if growth is None:
        growth = td.get("net_income_cagr_3y_pct")
    growth = float(growth) if growth is not None else 8.0
    fair_pe = _clamp(8 + max(growth, 0) * 0.35, 8, 22)
    fair_value = round(normalized_eps * fair_pe, 2)
    mos = round((fair_value / cmp - 1) * 100, 1) if cmp else None
    earnings_yield = round((normalized_eps / cmp * 100), 2) if cmp else None
    return {
        "applicable": True,
        "method": "Normalized P/E",
        "normalized_eps": round(normalized_eps, 2),
        "fair_pe": round(fair_pe, 2),
        "fair_value_per_share": fair_value,
        "margin_of_safety_pct": mos,
        "current_earnings_yield_pct": earnings_yield,
    }


def _compute_valuation_profile(td: Dict[str, Any]) -> Dict[str, Any]:
    method = _select_valuation_method(td)
    if method == "pb_roe":
        primary = _compute_pb_roe_value(td)
        reverse = {
            "applicable": primary.get("applicable", False),
            "method": "Reverse P/B-ROE",
            "implied_sustainable_roe_pct": primary.get("implied_sustainable_roe_pct"),
        }
    elif method == "normalized_pe":
        primary = _compute_normalized_pe_value(td)
        reverse = _compute_reverse_dcf(td)
    else:
        primary = _compute_dcf_value(td)
        reverse = _compute_reverse_dcf(td)

    verdict = "Low-confidence valuation"
    ref_value = (
        (primary.get("per_share") or {}).get("base")
        if primary.get("method") == "DCF"
        else primary.get("fair_value_per_share")
    )
    cmp = td.get("cmp") or 0
    if ref_value and cmp:
        gap = ref_value / cmp - 1
        if gap >= 0.2:
            verdict = "Undervalued"
        elif gap <= -0.15:
            verdict = "Overvalued"
        else:
            verdict = "Fairly valued"

    return {
        "primary_method": primary.get("method") if primary.get("applicable") else method.upper(),
        "primary": primary,
        "reverse": reverse,
        "verdict": verdict,
        "confidence": "Medium" if primary.get("applicable") else "Low",
    }


def _fetch_candidate_news(company_name: str, ticker: str) -> List[str]:
    clean_ticker = re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)
    query = urllib.parse.quote(f'"{company_name}" OR "{clean_ticker}" stock')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        cutoff = datetime.now() - timedelta(hours=_CANDIDATE_NEWS_LOOKBACK_HOURS)
        headlines: List[str] = []
        for item in root.findall(".//item")[:20]:
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
            headlines.append(title)
        return headlines
    except Exception as exc:
        logger.debug("Candidate news fetch failed for %s: %s", ticker, exc)
        return []


def _fetch_recent_headlines(company_name: str, ticker: str, lookback_hours: int = _CANDIDATE_NEWS_LOOKBACK_HOURS) -> List[str]:
    clean_ticker = re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)
    query = urllib.parse.quote(f'"{company_name}" OR "{clean_ticker}" stock')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        headlines: List[str] = []
        for item in root.findall(".//item")[:12]:
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
            headlines.append(title)
        return headlines
    except Exception as exc:
        logger.debug("Recent headline fetch failed for %s: %s", ticker, exc)
        return []


def _check_candidate_events_and_news(ticker: str) -> List[str]:
    risks: List[str] = []
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
        today = date.today()

        cal = yf_ticker.calendar
        if cal is not None:
            earnings_dates: List[date] = []
            if isinstance(cal, dict):
                raw_dates = cal.get("Earnings Date")
                if raw_dates:
                    earnings_dates = raw_dates if isinstance(raw_dates, list) else [raw_dates]
            else:
                for col in cal.columns:
                    earnings_dates.append(col.date() if hasattr(col, "date") else col)
            for ed in earnings_dates:
                if isinstance(ed, datetime):
                    ed = ed.date()
                if isinstance(ed, date):
                    days_away = (ed - today).days
                    if 0 <= days_away <= 14:
                        risks.append(f"earnings event in {days_away} day(s) on {ed}")
                        break

        ex_div_ts = info.get("exDividendDate")
        if ex_div_ts:
            try:
                ex_div = date.fromtimestamp(ex_div_ts)
                days_away = (ex_div - today).days
                if 0 <= days_away <= 7:
                    risks.append(f"ex-dividend date is close on {ex_div}")
            except Exception:
                pass

        company_name = info.get("shortName") or info.get("longName") or ticker
        headlines = _fetch_candidate_news(company_name, ticker)
        for headline in headlines:
            lowered = headline.lower()
            for keyword in _NEWS_RISK_KEYWORDS:
                if keyword in lowered:
                    risks.append(f"recent headline risk: {headline}")
                    break
    except Exception as exc:
        logger.debug("Candidate events/news check failed for %s: %s", ticker, exc)
    return risks[:3]


def _assess_forward_risk(
    ticker: str,
    quote: Dict[str, Any],
    td: Dict[str, Any],
    event_risks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    hard_reject_reasons: List[str] = []
    watchlist_reasons: List[str] = []

    eps_ttm = td.get("eps_ttm")
    eps_forward = td.get("eps_forward")
    pe_ttm = td.get("pe_ttm")
    pe_forward = td.get("pe_forward")
    revenue_growth = td.get("revenue_growth_yoy")
    earnings_growth = td.get("earnings_growth_yoy")
    analyst_rating = _parse_analyst_rating(quote.get("averageAnalystRating"))
    ccr = td.get("cash_conversion_ratio")
    screener_cons = td.get("screener_data", {}).get("cons") or []
    cons_text = " ".join(str(item).lower() for item in screener_cons)

    if earnings_growth is not None and earnings_growth < 0:
        hard_reject_reasons.append(f"earnings growth is negative at {earnings_growth:.1f}%")
    if revenue_growth is not None and revenue_growth < 0:
        watchlist_reasons.append(f"revenue growth is negative at {revenue_growth:.1f}%")

    if eps_ttm is not None and eps_forward is not None and eps_ttm > 0:
        if eps_forward < eps_ttm * 0.9:
            hard_reject_reasons.append(
                f"forward EPS ₹{eps_forward:.2f} is materially below trailing EPS ₹{eps_ttm:.2f}"
            )
        elif eps_forward < eps_ttm:
            watchlist_reasons.append(
                f"forward EPS ₹{eps_forward:.2f} is below trailing EPS ₹{eps_ttm:.2f}"
            )

    if (
        pe_ttm and pe_forward and pe_ttm > 0 and pe_forward > pe_ttm * 1.2
        and ((earnings_growth is not None and earnings_growth < 10) or
             (revenue_growth is not None and revenue_growth < 10))
    ):
        watchlist_reasons.append(
            f"forward P/E {pe_forward:.1f} is stretched vs trailing P/E {pe_ttm:.1f} without enough growth support"
        )

    if analyst_rating is not None and analyst_rating >= 3.5:
        hard_reject_reasons.append(f"analyst rating is weak at {analyst_rating:.1f}")
    elif analyst_rating is not None and analyst_rating >= 3.0:
        watchlist_reasons.append(f"analyst rating is cautious at {analyst_rating:.1f}")

    matched_cons = [phrase for phrase in _FORWARD_RISK_CONS_PHRASES if phrase in cons_text]
    if matched_cons:
        hard_reject_reasons.append(
            "Screener.in cons flag forward risk: " + ", ".join(sorted(set(matched_cons))[:3])
        )

    if ccr is not None and ccr < 0.8:
        watchlist_reasons.append(f"cash conversion ratio is weak at {ccr:.2f}")

    forensic_flags = td.get("forensic_flags") or []
    if len(forensic_flags) >= 2:
        watchlist_reasons.append(f"{len(forensic_flags)} forensic flags need review")

    for risk in event_risks or []:
        watchlist_reasons.append(risk)

    if hard_reject_reasons:
        status = "rejected"
    elif watchlist_reasons:
        status = "watchlist"
    else:
        status = "eligible"

    return {
        "ticker": ticker,
        "status": status,
        "hard_reject_reasons": hard_reject_reasons,
        "watchlist_reasons": watchlist_reasons,
        "all_reasons": hard_reject_reasons + watchlist_reasons,
    }


def _summarize_forward_outlook(
    ticker: str,
    td: Dict[str, Any],
    recent_notes: Optional[List[str]] = None,
    headlines: Optional[List[str]] = None,
    gate_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score = 0
    evidence: List[str] = []
    cautions: List[str] = []

    revenue_growth = td.get("revenue_growth_yoy")
    earnings_growth = td.get("earnings_growth_yoy")
    eps_ttm = td.get("eps_ttm")
    eps_forward = td.get("eps_forward")
    pe_ttm = td.get("pe_ttm")
    pe_forward = td.get("pe_forward")
    peg = td.get("peg")
    roe = td.get("roe")
    op_margin = td.get("operating_margin")
    ccr = td.get("cash_conversion_ratio")
    nde = td.get("net_debt_to_ebitda")
    forensic_flags = td.get("forensic_flags") or []
    screener = td.get("screener_data") or {}

    if revenue_growth is not None:
        if revenue_growth >= 15:
            score += 2
            evidence.append(f"revenue growth is healthy at {revenue_growth:.1f}%")
        elif revenue_growth >= 8:
            score += 1
            evidence.append(f"revenue growth is positive at {revenue_growth:.1f}%")
        elif revenue_growth < 0:
            score -= 1
            cautions.append(f"revenue growth is negative at {revenue_growth:.1f}%")

    if earnings_growth is not None:
        if earnings_growth >= 15:
            score += 2
            evidence.append(f"earnings growth is strong at {earnings_growth:.1f}%")
        elif earnings_growth >= 8:
            score += 1
            evidence.append(f"earnings growth is positive at {earnings_growth:.1f}%")
        elif earnings_growth < 0:
            score -= 2
            cautions.append(f"earnings growth is negative at {earnings_growth:.1f}%")

    if eps_ttm and eps_forward:
        if eps_forward >= eps_ttm * 1.1:
            score += 2
            evidence.append(f"forward EPS ₹{eps_forward:.2f} is above trailing EPS ₹{eps_ttm:.2f}")
        elif eps_forward > eps_ttm:
            score += 1
            evidence.append(f"forward EPS ₹{eps_forward:.2f} is ahead of trailing EPS ₹{eps_ttm:.2f}")
        elif eps_forward < eps_ttm * 0.9:
            score -= 2
            cautions.append(f"forward EPS ₹{eps_forward:.2f} is materially below trailing EPS ₹{eps_ttm:.2f}")
        elif eps_forward < eps_ttm:
            score -= 1
            cautions.append(f"forward EPS ₹{eps_forward:.2f} is below trailing EPS ₹{eps_ttm:.2f}")

    if peg is not None and peg > 0:
        if peg <= 1.5:
            score += 1
            evidence.append(f"PEG {peg:.2f} supports GARP-style compounding")
        elif peg >= 2.5:
            score -= 1
            cautions.append(f"PEG {peg:.2f} looks expensive for the growth on offer")

    if pe_ttm and pe_forward and pe_forward < pe_ttm:
        score += 1
        evidence.append(f"forward P/E {pe_forward:.1f} is below trailing P/E {pe_ttm:.1f}")
    elif pe_ttm and pe_forward and pe_forward > pe_ttm * 1.2:
        score -= 1
        cautions.append(f"forward P/E {pe_forward:.1f} is stretched vs trailing P/E {pe_ttm:.1f}")

    if roe is not None and roe >= 15:
        score += 1
        evidence.append(f"ROE remains solid at {roe:.1f}%")
    if op_margin is not None and op_margin >= 15:
        score += 1
        evidence.append(f"operating margin at {op_margin:.1f}% suggests some durability")
    elif op_margin is not None and op_margin < 8:
        score -= 1
        cautions.append(f"operating margin at {op_margin:.1f}% leaves less room for execution misses")

    if ccr is not None and ccr >= 0.9:
        score += 1
        evidence.append(f"cash conversion ratio {ccr:.2f} supports earnings quality")
    elif ccr is not None and ccr < 0.8:
        score -= 1
        cautions.append(f"cash conversion ratio {ccr:.2f} weakens forward confidence")

    if nde is not None and nde > 3:
        score -= 1
        cautions.append(f"net debt / EBITDA at {nde:.2f}x raises balance-sheet risk")
    elif nde is not None and nde <= 1.5:
        score += 1
        evidence.append(f"net debt / EBITDA at {nde:.2f}x leaves balance-sheet flexibility")

    if len(forensic_flags) >= 2:
        score -= 2
        cautions.append(f"{len(forensic_flags)} forensic flags weaken long-term compounding confidence")

    pros = screener.get("pros") or []
    cons = screener.get("cons") or []
    if pros:
        score += 1
        evidence.append(f"Screener.in pros include: {pros[0]}")
    if cons:
        score -= 1
        cautions.append(f"Screener.in cons include: {cons[0]}")

    positive_news = []
    negative_news = []
    for title in headlines or []:
        lower = title.lower()
        if any(keyword in lower for keyword in _FORWARD_POSITIVE_HEADLINE_KEYWORDS):
            positive_news.append(title)
        if any(keyword in lower for keyword in _FORWARD_NEGATIVE_HEADLINE_KEYWORDS):
            negative_news.append(title)
    if positive_news:
        score += 1
        evidence.append(f"recent headline support: {positive_news[0]}")
    if negative_news:
        score -= 1
        cautions.append(f"recent headline risk: {negative_news[0]}")

    matched_note_risks = []
    for note in recent_notes or []:
        lower = note.lower()
        for keyword in _FORWARD_NOTE_RISK_KEYWORDS:
            if keyword in lower:
                matched_note_risks.append(keyword)
    if matched_note_risks:
        cautions.append("stored notes contain active risk triggers: " + ", ".join(sorted(set(matched_note_risks))[:4]))

    if gate_assessment:
        status = gate_assessment.get("status")
        if status == "eligible":
            score += 1
            evidence.append("deterministic gate considers the setup eligible for fresh capital")
        elif status == "watchlist":
            score -= 1
            cautions.extend(gate_assessment.get("all_reasons")[:2])
        elif status == "rejected":
            score -= 3
            cautions.extend(gate_assessment.get("all_reasons")[:3])

    evidence = evidence[:4]
    cautions = cautions[:4]
    evidence_count = len(evidence) + len(cautions)
    if evidence_count <= 1:
        outlook = "High Uncertainty"
        xirr_band = "Unclear"
    elif score >= 5:
        outlook = "High Compounding Potential"
        xirr_band = ">20%"
    elif score >= 2:
        outlook = "Moderate Compounding Potential"
        xirr_band = "15–20%"
    elif score >= 0:
        outlook = "Low Forward Return Potential"
        xirr_band = "10–15%"
    else:
        outlook = "High Uncertainty"
        xirr_band = "<10%"

    return {
        "ticker": ticker,
        "outlook": outlook,
        "xirr_band": xirr_band,
        "score": score,
        "evidence": evidence,
        "cautions": cautions,
        "headlines": (headlines or [])[:3],
    }


def _compute_forensic_flags(d: Dict, fin: List[Dict], bs: List[Dict], cf: List[Dict]) -> List[str]:
    """
    Compute red-flag signals per GrahamPrompt Part E (Shenanigans review).
    Returns a list of flag strings, empty if clean.
    """
    flags: List[str] = []

    # 1. Cash conversion < 0.8 (OCF significantly below Net Income)
    ccr = d.get("cash_conversion_ratio")
    if ccr is not None and ccr < 0.8:
        flags.append(
            f"⚠️ Cash conversion ratio {ccr:.2f} < 0.8 — earnings may be ahead of cash reality"
        )

    # 2. Negative FCF while reporting positive Net Income
    if fin and cf:
        ni_latest  = fin[0].get("net_income") or 0
        fcf_latest = cf[0].get("free_cash_flow") or 0
        if ni_latest > 0 and fcf_latest < 0:
            flags.append(
                f"⚠️ Positive net income (₹{ni_latest:.0f} Cr) but negative FCF (₹{fcf_latest:.0f} Cr)"
            )

    # 3. Revenue declining while net income growing (margin story may not persist)
    rev_cagr = d.get("revenue_cagr_3y_pct")
    ni_cagr  = d.get("net_income_cagr_3y_pct")
    if rev_cagr is not None and ni_cagr is not None:
        if rev_cagr < 0 and ni_cagr > 0:
            flags.append(
                f"⚠️ Revenue CAGR {rev_cagr:.1f}% negative but Net Income CAGR {ni_cagr:.1f}% positive — unsustainable margin expansion?"
            )

    # 4. Rapidly rising accounts receivable relative to revenue
    if len(bs) >= 2 and len(fin) >= 2:
        ar_new = bs[0].get("accounts_receivable") or 0
        ar_old = bs[1].get("accounts_receivable") or 0
        rev_new = fin[0].get("total_revenue") or 0
        rev_old = fin[1].get("total_revenue") or 0
        if ar_old and rev_old and rev_new:
            ar_growth    = (ar_new - ar_old) / ar_old * 100 if ar_old > 0 else 0
            rev_growth   = (rev_new - rev_old) / rev_old * 100 if rev_old > 0 else 0
            if ar_growth > rev_growth + 20:  # receivables growing 20pp faster than revenue
                flags.append(
                    f"⚠️ Receivables grew {ar_growth:.1f}% vs revenue grew {rev_growth:.1f}% — possible revenue recognition risk"
                )

    # 5. High debt / equity
    de = d.get("debt_to_equity")
    if de is not None and de > 100:  # yfinance reports D/E as %, not ratio for some tickers
        de_display = de / 100 if de > 10 else de
        if de_display > 1.5:
            flags.append(f"⚠️ Debt/Equity {de_display:.2f}x — elevated leverage")

    # 6. Negative or thin interest coverage
    ic = d.get("interest_coverage")
    if ic is not None and ic < 3:
        flags.append(f"⚠️ Interest coverage {ic:.1f}x — thin debt-service buffer")

    # 7. Net debt issuance growing fast (proxy for increasing leverage)
    if cf:
        net_debt_chg = cf[0].get("net_issuance_payments_of_debt") or 0
        if net_debt_chg > 200:   # ₹200 Cr net new debt in latest year
            flags.append(
                f"⚠️ Net new debt raised ₹{net_debt_chg:.0f} Cr in latest year — leverage rising"
            )

    # 8. Net debt > 3× EBITDA
    nd_ebitda = d.get("net_debt_to_ebitda")
    if nd_ebitda is not None and nd_ebitda > 3:
        flags.append(f"⚠️ Net Debt / EBITDA {nd_ebitda:.1f}x — leverage above comfort zone")

    return flags


# ──────────────────────────────────────────────
# Context-building blocks
# ──────────────────────────────────────────────

def _build_positions_table(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
    today: date,
    cash_by_portfolio: Optional[Dict[str, float]] = None,
) -> str:
    """
    Full position table with all GrahamPrompt-required per-position fields:
    portfolio %, holding period classification, LTCG status,
    XIRR proxy, allocation rank.
    """
    lines: List[str] = ["## Current Holdings — Detailed Position Table\n"]

    cash_map = cash_by_portfolio or {}

    for portfolio, tickers in positions_by_portfolio.items():
        # Portfolio-level totals for allocation %
        port_current_val = sum(
            pos["net_shares"] * (ticker_data.get(tk, {}).get("cmp") or 0)
            for tk, pos in tickers.items()
        )
        port_invested = sum(
            pos["total_invested"] * ticker_data.get(tk, {}).get("fx_rate_to_inr", 1.0)
            for tk, pos in tickers.items()
        )
        cash_balance  = cash_map.get(portfolio, 0.0)
        port_total    = port_current_val + cash_balance  # denominator for all port%

        lines.append(
            f"### Portfolio: {portfolio}\n"
            f"Invested: ₹{port_invested:,.0f} | Equity Current: ₹{port_current_val:,.0f} | "
            f"Cash: ₹{cash_balance:,.0f} | Total Value: ₹{port_total:,.0f} | "
            f"Unrealised P&L: ₹{port_current_val - port_invested:,.0f} "
            f"({((port_current_val - port_invested) / port_invested * 100) if port_invested else 0:+.1f}%)\n"
        )

        # Sort by current value descending (largest position first)
        sorted_tickers = sorted(
            tickers.items(),
            key=lambda kv: kv[1]["net_shares"] * (ticker_data.get(kv[0], {}).get("cmp") or 0),
            reverse=True,
        )

        lines.append(
            "| # | Ticker | Company | Sector Bucket | Shares | Avg Cost | CMP | "
            "Invested ₹ | Current ₹ | Port% | Unreal P&L% | XIRR Proxy% | "
            "Holding Days | HP Classification | LTCG Status | Forensic Flags |"
        )
        lines.append("|---|--------|---------|---------------|--------|----------|-----|"
                     "------------|-----------|-------|-------------|-------------|"
                     "--------------|-------------------|-------------|----------------|")

        for rank, (ticker, pos) in enumerate(sorted_tickers, 1):
            td      = ticker_data.get(ticker, {})
            cmp     = td.get("cmp") or 0           # always INR
            fx      = td.get("fx_rate_to_inr", 1.0)
            avg_native = pos["avg_cost"]
            avg     = avg_native * fx               # avg cost in INR
            net     = pos["net_shares"]
            cur_val = net * cmp
            inv_val = pos["total_invested"] * fx    # INR
            unreal_pct = ((cmp - avg) / avg * 100) if avg > 0 and cmp > 0 else 0.0
            port_pct   = (cur_val / port_total * 100) if port_total > 0 else 0.0

            # Holding period (from first buy date)
            earliest_buy = min(pos["buy_dates"]) if pos["buy_dates"] else None
            latest_buy   = max(pos["buy_dates"]) if pos["buy_dates"] else None
            holding_days = (today - date.fromisoformat(earliest_buy)).days if earliest_buy else 0
            hp_class     = _classify_holding_period(holding_days)
            ltcg         = _ltcg_status(earliest_buy, today) if earliest_buy else "Unknown"

            # XIRR proxy: simple CAGR from weighted-average buy date to today
            # Compare native-currency avg vs native CMP to strip FX noise
            cmp_native = td.get("cmp_native") or 0
            xirr_proxy = None
            if pos["all_buy_lots"] and cmp_native > 0 and avg_native > 0:
                total_cost = sum(s * c for _, s, c in pos["all_buy_lots"])
                wav_ts = sum(
                    (date.fromisoformat(d) - date(1970, 1, 1)).days * s * c
                    for d, s, c in pos["all_buy_lots"]
                ) / total_cost if total_cost else 0
                wav_date = date(1970, 1, 1) + timedelta(days=wav_ts)
                years_held = (today - wav_date).days / 365.25
                if years_held > 0:
                    xirr_proxy = _cagr(avg_native, cmp_native, years_held)

            flags = td.get("forensic_flags") or []
            flags_str = "; ".join(flags) if flags else "None"
            xirr_str = f"{xirr_proxy:+.1f}%" if xirr_proxy is not None else "N/A"

            currency_label = td.get("currency", "INR")
            avg_display = f"₹{avg:,.2f}" if currency_label == "INR" else f"{currency_label} {avg_native:,.2f} (₹{avg:,.2f})"
            cmp_display = f"₹{cmp:,.2f}" if currency_label == "INR" else f"{currency_label} {cmp_native:,.2f} (₹{cmp:,.2f})"
            lines.append(
                f"| {rank} | {ticker} | {td.get('company_name', ticker)[:20]} | "
                f"{td.get('sector_bucket', 'N/A')} | {net:.2f} | "
                f"{avg_display} | {cmp_display} | "
                f"₹{inv_val:,.0f} | ₹{cur_val:,.0f} | "
                f"{port_pct:.1f}% | {unreal_pct:+.1f}% | {xirr_str} | "
                f"{holding_days} | {hp_class} | {ltcg} | {flags_str} |"
            )

        if cash_balance > 0:
            cash_pct = (cash_balance / port_total * 100) if port_total > 0 else 0.0
            lines.append(
                f"| — | CASH | Cash / Liquid | Cash / Liquid | — | — | ₹1.00 | "
                f"₹{cash_balance:,.0f} | ₹{cash_balance:,.0f} | "
                f"{cash_pct:.1f}% | +0.0% | N/A | — | — | — | None |"
            )
        lines.append("")

    return "\n".join(lines)


def _build_position_notes_section(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
) -> str:
    lines: List[str] = ["## Transaction Notes Context\n"]
    any_notes = False

    for portfolio, tickers in positions_by_portfolio.items():
        portfolio_lines: List[str] = []
        for ticker, pos in tickers.items():
            notes = pos.get("notes") or []
            if not notes:
                continue
            any_notes = True
            portfolio_lines.append(f"### {portfolio} — {ticker}")
            for note in notes[:5]:
                note_date = note.get("date") or "Unknown date"
                note_type = note.get("type") or "Txn"
                portfolio_lines.append(f"- {note_date} [{note_type}] {note.get('note')}")
            portfolio_lines.append("")
        if portfolio_lines:
            lines.extend(portfolio_lines)

    if not any_notes:
        lines.append("_No transaction notes recorded in the portfolio transactions table._")
    return "\n".join(lines)


def _build_reconciliation_checks(transactions: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["## Data Reconciliation Checks\n"]
    cash_rows = []
    trade_rows = []
    for tx in transactions:
        ticker = (tx.get("Ticker") or "").strip().upper()
        if ticker == _CASH_TICKER:
            cash_rows.append({
                "portfolio": (tx.get("Portfolio") or "Default").strip(),
                "date": (tx.get("Date") or "")[:10],
                "type": (tx.get("TransactionType") or "").strip(),
                "amount": round(float(tx.get("NoOfShares") or 0) * float(tx.get("CostPerShare") or 0), 2),
                "used": False,
            })
        elif ticker:
            tx_type = (tx.get("TransactionType") or "").strip()
            if tx_type not in {"Buy", "Sell"}:
                continue
            trade_rows.append({
                "ticker": ticker,
                "portfolio": (tx.get("Portfolio") or "Default").strip(),
                "date": (tx.get("Date") or "")[:10],
                "cash_type": "Withdraw" if tx_type == "Buy" else "Deposit",
                "amount": round(float(tx.get("NoOfShares") or 0) * float(tx.get("CostPerShare") or 0), 2),
            })

    missing_matches: List[str] = []
    for trade in trade_rows:
        matched = False
        for cash in cash_rows:
            if cash["used"]:
                continue
            if (
                cash["portfolio"] == trade["portfolio"]
                and cash["date"] == trade["date"]
                and cash["type"] in {trade["cash_type"], "Withdrawal" if trade["cash_type"] == "Withdraw" else trade["cash_type"]}
                and math.isclose(cash["amount"], trade["amount"], rel_tol=0.0, abs_tol=0.5)
            ):
                cash["used"] = True
                matched = True
                break
        if not matched:
            missing_matches.append(
                f"{trade['portfolio']} {trade['date']} {trade['ticker']} requires {trade['cash_type']} ₹{trade['amount']:,.2f}"
            )

    unmatched_cash = [
        c for c in cash_rows
        if not c["used"] and c["type"] in {"Deposit", "Withdraw", "Withdrawal"}
    ]

    if missing_matches:
        lines.append("### Missing Trade-Cash Matches")
        for item in missing_matches[:10]:
            lines.append(f"- {item}")
    else:
        lines.append("- All Buy/Sell rows have matching CASH entries within tolerance.")

    if unmatched_cash:
        lines.append("\n### Unmatched CASH Rows")
        for cash in unmatched_cash[:10]:
            lines.append(
                f"- {cash['portfolio']} {cash['date']} {cash['type']} ₹{cash['amount']:,.2f} "
                "(may be external funding, or may need manual review)"
            )
    else:
        lines.append("\n- No unmatched CASH rows detected.")

    lines.append("\nIf a metric is missing or inconsistent, state `Insufficient evidence` instead of inferring or inventing it.\n")
    return "\n".join(lines)


def _build_duplicate_exposure_table(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
    total_portfolio_value: float,
    cash_by_portfolio: Optional[Dict[str, float]] = None,
) -> str:
    """
    GrahamPrompt Level 4: aggregate exposure for stocks held in multiple portfolios.
    """
    from collections import defaultdict
    cross: Dict[str, Dict] = defaultdict(lambda: {
        "portfolios": [], "total_shares": 0.0, "total_invested": 0.0,
    })

    for portfolio, tickers in positions_by_portfolio.items():
        for ticker, pos in tickers.items():
            fx = ticker_data.get(ticker, {}).get("fx_rate_to_inr", 1.0)
            cross[ticker]["portfolios"].append(portfolio)
            cross[ticker]["total_shares"]   += pos["net_shares"]
            cross[ticker]["total_invested"] += pos["total_invested"] * fx

    dupes = {t: v for t, v in cross.items() if len(v["portfolios"]) > 1}
    if not dupes:
        return "## Cross-Portfolio Duplicate Holdings\nNone — each ticker held in exactly one portfolio.\n"

    total_cash  = sum((cash_by_portfolio or {}).values())
    grand_total = total_portfolio_value + total_cash

    lines = [
        "## Cross-Portfolio Duplicate Holdings (GrahamPrompt Level 4)\n",
        "| Ticker | Company | Held In | Total Shares | Total Invested | Total Current | Total Portfolio% | Concentration Risk |",
        "|--------|---------|---------|-------------|---------------|--------------|-------------------|-------------------|",
    ]
    for ticker, info in dupes.items():
        td       = ticker_data.get(ticker, {})
        cmp      = td.get("cmp") or 0
        cur_val  = info["total_shares"] * cmp
        tot_pct  = (cur_val / grand_total * 100) if grand_total else 0
        risk     = "HIGH ⚠️" if tot_pct > _MAX_SINGLE_STOCK_PCT else ("MEDIUM" if tot_pct > 10 else "LOW")
        lines.append(
            f"| {ticker} | {td.get('company_name', ticker)[:20]} | "
            f"{', '.join(info['portfolios'])} | {info['total_shares']:.2f} | "
            f"₹{info['total_invested']:,.0f} | ₹{cur_val:,.0f} | "
            f"{tot_pct:.1f}% | {risk} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_position_sizing_violations(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
    cash_by_portfolio: Optional[Dict[str, float]] = None,
) -> str:
    """
    Flag positions violating GrahamPrompt Part H Step 2 sizing rules.
    Portfolio % denominator includes cash so sizing is not overstated.
    """
    cash_map = cash_by_portfolio or {}
    lines = [
        "## Position Sizing Violation Flags (GrahamPrompt Part H Step 2)\n",
        f"Rules: Max single stock {_MAX_SINGLE_STOCK_PCT}% | "
        f"Max low-conviction {_MAX_LOW_CONVICTION_PCT}% | "
        f"Max cyclical {_MAX_CYCLICAL_PCT}%\n",
    ]
    any_violation = False

    for portfolio, tickers in positions_by_portfolio.items():
        equity_val = sum(
            pos["net_shares"] * (ticker_data.get(tk, {}).get("cmp") or 0)
            for tk, pos in tickers.items()
        )
        port_val = equity_val + cash_map.get(portfolio, 0.0)
        if port_val == 0:
            continue
        for ticker, pos in tickers.items():
            td      = ticker_data.get(ticker, {})
            cmp     = td.get("cmp") or 0
            cur_val = pos["net_shares"] * cmp
            pct     = cur_val / port_val * 100
            flags   = []
            if pct > _MAX_SINGLE_STOCK_PCT:
                flags.append(f"Exceeds max single-stock cap ({_MAX_SINGLE_STOCK_PCT}%)")
            bucket = (td.get("sector_bucket") or "").lower()
            if any(x in bucket for x in ["metal", "energy", "cyclical"]) and pct > _MAX_CYCLICAL_PCT:
                flags.append(f"Cyclical stock exceeds {_MAX_CYCLICAL_PCT}% cap")
            if flags:
                any_violation = True
                lines.append(
                    f"- **{portfolio} / {ticker}**: {pct:.1f}% → {'; '.join(flags)}"
                )

    if not any_violation:
        lines.append("No sizing violations detected.")
    lines.append("")
    return "\n".join(lines)


def _build_sector_allocation(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
    total_value: float,
    cash_by_portfolio: Optional[Dict[str, float]] = None,
) -> str:
    """Sector allocation across all portfolios (GrahamPrompt Level 1)."""
    from collections import defaultdict
    sector_val: Dict[str, float] = defaultdict(float)

    for tickers in positions_by_portfolio.values():
        for ticker, pos in tickers.items():
            td      = ticker_data.get(ticker, {})
            cmp     = td.get("cmp") or 0
            bucket  = td.get("sector_bucket") or "Unknown"
            sector_val[bucket] += pos["net_shares"] * cmp

    total_cash = sum((cash_by_portfolio or {}).values())
    if total_cash > 0:
        sector_val["Cash / Liquid"] += total_cash

    grand_total = total_value + total_cash
    lines = [
        "## Overall Sector Allocation\n",
        "| Sector Bucket | Current Value (₹) | Portfolio % | Valuation Framework Used |",
        "|--------------|-------------------|-------------|--------------------------|",
    ]
    frameworks = {
        "Banks":                    "P/B, ROE, NIM, NPA, CASA, CAR",
        "Financials / NBFC":        "P/B, NIM, ALM risk, Loan-book quality",
        "Technology":               "FCF yield, Revenue quality, Buybacks, ROCE",
        "Pharma / Healthcare":      "ROCE, FCF, R&D, Regulatory risk, Export",
        "Energy / Utilities":       "P/E, P/B, Div yield, FCF, Cyclicality",
        "Metals / Chemicals":       "P/E, P/B, FCF, Balance sheet, Commodity price",
        "Manufacturing / Industrials": "ROCE, FCF, Capacity utilisation, Debt",
        "Consumer":                 "P/E, Revenue growth, Margin, Brand moat",
        "Real Estate":              "NAV, Debt, Cash flow, Pre-sales",
        "Cash / Liquid":            "Deploy to best-XIRR opportunity per Part K plan",
    }
    for bucket in sorted(sector_val, key=sector_val.get, reverse=True):
        val = sector_val[bucket]
        pct = (val / grand_total * 100) if grand_total else 0
        fw  = frameworks.get(bucket, "Standard P/E, P/B, ROE")
        lines.append(f"| {bucket} | ₹{val:,.0f} | {pct:.1f}% | {fw} |")

    lines.append("")
    return "\n".join(lines)


def _build_market_data_section(ticker_data: Dict[str, Dict]) -> str:
    """
    Per-ticker deep-dive block. Structured to match every GrahamPrompt Part:
    Part D (GARP), Part E (Shenanigans), Part F (sector rules), Part G (intrinsic value).
    """
    lines = ["## Per-Ticker Market & Fundamental Deep-Dive\n"]

    for ticker, td in ticker_data.items():
        if td.get("error"):
            lines.append(f"### {ticker}\n_Data fetch error: {td['error']}_\n")
            continue

        currency = td.get("currency", "INR")
        fx_note  = f" (native {currency}; monetary figures converted to ₹ at {td.get('fx_rate_to_inr', 1.0):.2f})" if currency != "INR" else ""
        lines.append(f"### {ticker} — {td.get('company_name', ticker)}")
        lines.append(f"**Sector Bucket:** {td.get('sector_bucket')} | "
                     f"**Exchange:** {td.get('exchange')} | "
                     f"**Industry:** {td.get('industry')}{fx_note}")
        lines.append(f"**Business:** {_fmt_text(td.get('business_summary'))}")
        lines.append("")

        valuation = td.get("valuation") or {}
        primary = valuation.get("primary") or {}
        reverse = valuation.get("reverse") or {}
        lines.append("**Valuation Engine:**")
        lines.append(
            f"- Method: {_fmt_text(valuation.get('primary_method'))} | Verdict: {_fmt_text(valuation.get('verdict'))} | Confidence: {_fmt_text(valuation.get('confidence'))}"
        )
        if primary.get("method") == "DCF" and primary.get("applicable"):
            per_share = primary.get("per_share") or {}
            lines.append(
                f"- DCF per-share value — Bear {_fmt_num(per_share.get('bear'), prefix='₹')} | Base {_fmt_num(per_share.get('base'), prefix='₹')} | Bull {_fmt_num(per_share.get('bull'), prefix='₹')} | "
                f"Margin of Safety: {_fmt_pct(primary.get('margin_of_safety_pct'))}"
            )
            lines.append(
                f"- DCF assumptions — Cost of Equity {_fmt_pct(primary.get('cost_of_equity_pct'))} | "
                f"Initial Growth {_fmt_pct(primary.get('initial_growth_pct'))} | Terminal Growth {_fmt_pct(primary.get('terminal_growth_pct'))}"
            )
        elif primary.get("method") == "P/B-ROE" and primary.get("applicable"):
            lines.append(
                f"- Fair value/share {_fmt_num(primary.get('fair_value_per_share'), prefix='₹')} | Justified P/B {_fmt_text(primary.get('justified_pb'))} | "
                f"Margin of Safety: {_fmt_pct(primary.get('margin_of_safety_pct'))}"
            )
            lines.append(
                f"- P/B-ROE inputs — Cost of Equity {_fmt_pct(primary.get('cost_of_equity_pct'))} | "
                f"Sustainable Growth {_fmt_pct(primary.get('growth_pct'))} | Implied Sustainable ROE at current P/B {_fmt_pct(primary.get('implied_sustainable_roe_pct'))}"
            )
        elif primary.get("method") == "Normalized P/E" and primary.get("applicable"):
            lines.append(
                f"- Fair value/share {_fmt_num(primary.get('fair_value_per_share'), prefix='₹')} | Fair P/E {_fmt_text(primary.get('fair_pe'))}x | "
                f"Normalized EPS {_fmt_num(primary.get('normalized_eps'), prefix='₹')} | Margin of Safety: {_fmt_pct(primary.get('margin_of_safety_pct'))}"
            )
            lines.append(
                f"- Earnings yield at CMP: {_fmt_pct(primary.get('current_earnings_yield_pct'))}"
            )
        else:
            lines.append(f"- Primary valuation unavailable: {_fmt_text(primary.get('reason'))}")

        if reverse.get("applicable"):
            if reverse.get("method") == "Reverse DCF":
                lines.append(
                    f"- Reverse DCF: CMP implies ~{_fmt_pct(reverse.get('implied_fcf_growth_5y_pct'))} 5Y FCF growth "
                    f"(CoE {_fmt_pct(reverse.get('cost_of_equity_pct'))}, terminal growth {_fmt_pct(reverse.get('terminal_growth_pct'))})"
                )
            elif reverse.get("method") == "Reverse P/B-ROE":
                lines.append(
                    f"- Reverse valuation: current P/B implies sustainable ROE of ~{_fmt_pct(reverse.get('implied_sustainable_roe_pct'))}"
                )
        else:
            lines.append(f"- Reverse valuation unavailable: {_fmt_text(reverse.get('reason'))}")

        lines.append("**Valuation Multiples:**")
        lines.append(
            f"- P/E TTM: {_fmt_text(td.get('pe_ttm'))} | P/E Fwd: {_fmt_text(td.get('pe_forward'))} | "
            f"P/B: {_fmt_text(td.get('pb'))} | P/S: {_fmt_text(td.get('ps'))} | EV/EBITDA: {_fmt_text(td.get('ev_ebitda'))} | PEG: {_fmt_text(td.get('peg'))}"
        )
        lines.append(
            f"- Book Value/Share: {_fmt_num(td.get('book_value'), prefix='₹')} | EPS TTM: {_fmt_num(td.get('eps_ttm'), prefix='₹')} | EPS Fwd: {_fmt_num(td.get('eps_forward'), prefix='₹')}"
        )
        lines.append(
            f"- Market Cap: {_fmt_num(td.get('market_cap_cr'), prefix='₹', suffix=' Cr')} | FCF Yield: {_fmt_pct(td.get('fcf_yield_pct'))}"
        )
        lines.append(
            f"- 52W High: {_fmt_num(td.get('52w_high'), prefix='₹')} | 52W Low: {_fmt_num(td.get('52w_low'), prefix='₹')} | "
            f"CMP vs 52W High: {_fmt_pct(round((td.get('cmp') / td.get('52w_high') - 1) * 100, 1) if td.get('cmp') and td.get('52w_high') else None)} | "
            f"CMP vs 52W Low: {_fmt_pct(round((td.get('cmp') / td.get('52w_low') - 1) * 100, 1) if td.get('cmp') and td.get('52w_low') else None)}"
        )
        lines.append(
            f"- 50D MA: {_fmt_num(td.get('50d_ma'), prefix='₹')} | 200D MA: {_fmt_num(td.get('200d_ma'), prefix='₹')} | Beta: {_fmt_text(td.get('beta'))}"
        )

        lines.append("\n**Profitability & Returns:**")
        lines.append(
            f"- ROE: {_fmt_pct(td.get('roe'))} | ROA: {_fmt_pct(td.get('roa'))}"
        )
        lines.append(
            f"- Profit Margin: {_fmt_pct(td.get('profit_margin'))} | Operating Margin: {_fmt_pct(td.get('operating_margin'))} | "
            f"Gross Margin: {_fmt_pct(td.get('gross_margin'))} | EBITDA Margin: {_fmt_pct(td.get('ebitda_margin'))}"
        )
        lines.append(
            f"- FCF Margin: {_fmt_pct(td.get('fcf_margin_pct'))} | Interest Coverage: {_fmt_ratio(td.get('interest_coverage'))}"
        )

        lines.append("\n**Growth Profile (GARP Criteria):**")
        lines.append(
            f"- Revenue YoY: {_fmt_pct(td.get('revenue_growth_yoy'))} | Earnings YoY: {_fmt_pct(td.get('earnings_growth_yoy'))}"
        )
        lines.append(
            f"- Forward EPS vs TTM EPS: {_fmt_num(td.get('eps_forward'), prefix='₹')} vs {_fmt_num(td.get('eps_ttm'), prefix='₹')} | "
            f"Forward P/E vs TTM P/E: {_fmt_text(td.get('pe_forward'))} vs {_fmt_text(td.get('pe_ttm'))}"
        )
        lines.append(
            f"- Revenue CAGR 3Y: {_fmt_pct(td.get('revenue_cagr_3y_pct'))} | 5Y: {_fmt_pct(td.get('revenue_cagr_5y_pct'))}"
        )
        lines.append(
            f"- Net Income CAGR 3Y: {_fmt_pct(td.get('net_income_cagr_3y_pct'))} | 5Y: {_fmt_pct(td.get('net_income_cagr_5y_pct'))}"
        )
        lines.append(
            f"- Price CAGR 5Y: {_fmt_pct(td.get('price_cagr_5y_pct'))} | 1Y Price Return: {_fmt_pct(td.get('price_return_1y_pct'))}"
        )
        lines.append(
            f"- PEG Ratio: {_fmt_text(td.get('peg'))} (GARP target: <1.5 for strong conviction)"
        )

        lines.append("\n**Cash Flow Quality (Shenanigans Screen):**")
        lines.append(
            f"- Operating CF: {_fmt_num(td.get('operating_cashflow_cr'), prefix='₹', suffix=' Cr')} | FCF: {_fmt_num(td.get('free_cash_flow_cr'), prefix='₹', suffix=' Cr')} | "
            f"Capex: {_fmt_num(td.get('capex_cr'), prefix='₹', suffix=' Cr')}"
        )
        lines.append(
            f"- Cash Conversion Ratio (OCF/Net Income): {_fmt_text(td.get('cash_conversion_ratio'))} "
            f"(≥0.8 = clean; <0.6 = serious concern)"
        )
        if td.get("annual_cashflows"):
            lines.append("- Annual Cash Flow Trend (₹ Cr):")
            for yr in td["annual_cashflows"]:
                lines.append(
                    f"  {yr['year']}: OCF {_fmt_text(yr.get('operating_cash_flow'))} Cr | "
                    f"FCF {_fmt_text(yr.get('free_cash_flow'))} Cr | "
                    f"Capex {_fmt_text(yr.get('capital_expenditure'))} Cr | "
                    f"Net Debt Chg {_fmt_text(yr.get('net_issuance_payments_of_debt'))} Cr | "
                    f"Dividends {_fmt_text(yr.get('cash_dividends_paid'))} Cr"
                )

        lines.append("\n**Balance Sheet Strength:**")
        lines.append(
            f"- Debt/Equity: {_fmt_text(td.get('debt_to_equity'))} | Current Ratio: {_fmt_text(td.get('current_ratio'))} | "
            f"Quick Ratio: {_fmt_text(td.get('quick_ratio'))}"
        )
        lines.append(
            f"- Total Debt: {_fmt_num(td.get('total_debt_cr'), prefix='₹', suffix=' Cr')} | Cash: {_fmt_num(td.get('cash_cr'), prefix='₹', suffix=' Cr')} | "
            f"Net Debt: {_fmt_num(td.get('net_debt_cr'), prefix='₹', suffix=' Cr')} | Net Debt/EBITDA: {_fmt_ratio(td.get('net_debt_to_ebitda'))}"
        )
        if td.get("annual_balance_sheet"):
            lines.append("- Balance Sheet Trend (₹ Cr):")
            for yr in td["annual_balance_sheet"]:
                lines.append(
                    f"  {yr['year']}: Receivables {_fmt_text(yr.get('accounts_receivable'))} | "
                    f"Inventory {_fmt_text(yr.get('inventory'))} | "
                    f"Equity {_fmt_text(yr.get('stockholders_equity'))} | "
                    f"LT Debt {_fmt_text(yr.get('long_term_debt'))}"
                )

        lines.append("\n**Annual P&L Trend (₹ Cr) — 5-Year History:**")
        if td.get("annual_financials"):
            for yr in td["annual_financials"]:
                lines.append(
                    f"  {yr['year']}: Revenue {_fmt_text(yr.get('total_revenue'))} | "
                    f"Gross Profit {_fmt_text(yr.get('gross_profit'))} | "
                    f"EBIT {_fmt_text(yr.get('ebit'))} | "
                    f"Net Income {_fmt_text(yr.get('net_income'))}"
                )
        else:
            lines.append("  _Insufficient evidence_")

        lines.append("\n**Dividend & Shareholder Returns:**")
        lines.append(
            f"- Dividend Yield: {_fmt_pct(td.get('dividend_yield'), decimals=2)} | Rate: {_fmt_num(td.get('dividend_rate'), prefix='₹')} | "
            f"Payout Ratio: {_fmt_pct(td.get('payout_ratio'))}"
        )

        lines.append("\n**Ownership & Governance:**")
        lines.append(
            f"- Insider: {_fmt_pct(td.get('insider_ownership_pct'))} | Institutional: {_fmt_pct(td.get('institutional_ownership_pct'))}"
        )
        lines.append(
            f"- Shares Outstanding: {_fmt_num(td.get('shares_outstanding_cr'), decimals=4, suffix=' Cr')} | "
            f"Short Ratio: {_fmt_text(td.get('short_ratio'))} | Short % Float: {_fmt_pct(td.get('shares_short_pct_float'))}"
        )

        flags = td.get("forensic_flags") or []
        lines.append("\n**Forensic Accounting Signals (Part E Screen):**")
        if flags:
            for f in flags:
                lines.append(f"  {f}")
        else:
            lines.append("  ✅ No automated forensic flags detected")

        # Screener.in qualitative signals
        screener = td.get("screener_data") or {}
        pros = screener.get("pros") or []
        cons = screener.get("cons") or []
        if pros or cons:
            lines.append("\n**Screener.in Qualitative Signals (machine-generated from financials):**")
            for p in pros:
                lines.append(f"  ✅ {p}")
            for c in cons:
                lines.append(f"  ❌ {c}")
        else:
            lines.append("\n**Screener.in:** _Insufficient evidence_")

        # Sector-specific note (Part F)
        bucket = td.get("sector_bucket", "")
        if bucket == "Banks":
            lines.append(
                "\n**Sector Note (Banks — Part F):** "
                "Evaluate: P/B, ROE, NIM, Gross NPA%, Net NPA%, Provision Coverage, "
                "CASA ratio, Credit Cost, Loan Growth Quality, CAR. "
                "Do NOT use Current Ratio or Graham Number."
            )
        elif "NBFC" in bucket:
            lines.append(
                "\n**Sector Note (NBFC — Part F):** "
                "Evaluate: Cost of Funds, Spread, NIM, Asset Quality, ALM risk, "
                "Provisioning conservatism, Regulatory risk."
            )
        elif bucket == "Technology":
            lines.append(
                "\n**Sector Note (Technology — Part F):** "
                "Evaluate: FCF generation, Recurring revenue quality, "
                "Margin durability, Buybacks, Pricing power, Customer retention."
            )
        elif bucket == "Pharma / Healthcare":
            lines.append(
                "\n**Sector Note (Pharma — Part F):** "
                "Evaluate: ROCE, FCF conversion, Regulatory risk, Export dependence, "
                "Product concentration, R&D pipeline, Working capital."
            )
        elif bucket in ("Metals / Chemicals", "Energy / Utilities"):
            lines.append(
                "\n**Sector Note (Cyclicals — Part F):** "
                "Evaluate: P/E, P/B, Dividend yield, FCF, Balance sheet. "
                "Distinguish: cheap-cyclical vs structurally-impaired vs genuinely-mispriced."
            )

        lines.append("\n" + "─" * 60 + "\n")

    return "\n".join(lines)


def _build_forward_outlook_section(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
) -> str:
    lines = ["## Forward Outlook — Current Holdings\n"]
    for portfolio, tickers in positions_by_portfolio.items():
        portfolio_lines: List[str] = []
        for ticker, pos in tickers.items():
            td = ticker_data.get(ticker, {})
            company_name = td.get("company_name") or ticker
            notes = [n.get("note", "") for n in (pos.get("notes") or [])[:3] if n.get("note")]
            headlines = _fetch_recent_headlines(company_name, ticker)
            outlook = _summarize_forward_outlook(ticker, td, recent_notes=notes, headlines=headlines)
            portfolio_lines.append(
                f"### {portfolio} — {ticker}\n"
                f"- Forward class: **{outlook['outlook']}** | Expected XIRR band: **{outlook['xirr_band']}**\n"
                f"- Positive evidence: {'; '.join(outlook['evidence']) if outlook['evidence'] else 'Limited hard forward positives available'}\n"
                f"- Key cautions: {'; '.join(outlook['cautions']) if outlook['cautions'] else 'No major forward caution flags from current data'}\n"
                f"- Recent headlines: {' | '.join(outlook['headlines']) if outlook['headlines'] else 'No recent high-signal headlines captured'}"
            )
            if notes:
                portfolio_lines.append(f"- Stored thesis notes: {' | '.join(notes[:2])}")
            portfolio_lines.append("")
        if portfolio_lines:
            lines.extend(portfolio_lines)
    return "\n".join(lines)


def _build_candidate_forward_outlook_section(
    candidate_rows: List[Dict[str, Any]],
) -> str:
    lines = ["## Forward Outlook — Candidate Recommendations\n"]
    if not candidate_rows:
        lines.append("_No screened candidates available for forward assessment._")
        return "\n".join(lines)

    for row in candidate_rows:
        ticker = row["ticker"]
        td = row["ticker_data"]
        assessment = row["assessment"]
        quote = row.get("quote") or {}
        headlines = _fetch_recent_headlines(td.get("company_name") or ticker, ticker)
        outlook = _summarize_forward_outlook(
            ticker,
            td,
            headlines=headlines,
            gate_assessment=assessment,
        )
        analyst = _parse_analyst_rating(quote.get("averageAnalystRating"))
        lines.append(
            f"### {ticker} — {td.get('company_name', ticker)}\n"
            f"- Gate status: **{assessment.get('status', 'unknown').title()}** | Forward class: **{outlook['outlook']}** | Expected XIRR band: **{outlook['xirr_band']}**\n"
            f"- Positive evidence: {'; '.join(outlook['evidence']) if outlook['evidence'] else 'Limited hard forward positives available'}\n"
            f"- Key cautions: {'; '.join(outlook['cautions']) if outlook['cautions'] else 'No major forward caution flags from current data'}\n"
            f"- Supporting data: Analyst rating {_fmt_text(analyst)} | PEG {_fmt_text(td.get('peg'))} | Revenue YoY {_fmt_pct(td.get('revenue_growth_yoy'))} | Earnings YoY {_fmt_pct(td.get('earnings_growth_yoy'))}\n"
            f"- Recent headlines: {' | '.join(outlook['headlines']) if outlook['headlines'] else 'No recent high-signal headlines captured'}\n"
        )
    return "\n".join(lines)


def _build_recent_transaction_review(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    today: date,
) -> str:
    """
    GrahamPrompt 'Recent Transaction Awareness' table.
    Lists all positions bought within 90 days requiring extra care before Trim/Exit.
    """
    lines = [
        "## Recent Transaction Awareness (GrahamPrompt Rule — Do NOT exit prematurely)\n",
        "| Portfolio | Ticker | Latest Buy | Days Held | HP Class | "
        "LTCG Status | Guard Rule |",
        "|-----------|--------|-----------|-----------|----------|"
        "-------------|------------|",
    ]
    any_recent = False

    for portfolio, tickers in positions_by_portfolio.items():
        for ticker, pos in tickers.items():
            latest_buy = max(pos["buy_dates"]) if pos["buy_dates"] else None
            if not latest_buy:
                continue
            days = (today - date.fromisoformat(latest_buy)).days
            if days > _HP_RECENT:
                continue
            any_recent = True
            hp = _classify_holding_period(days)
            ltcg = _ltcg_status(min(pos["buy_dates"]), today)
            guard = (
                "⛔ Do not exit unless: accounting red flag, governance crisis, or thesis broken"
                if days <= _HP_VERY_RECENT
                else "⚠️ Prefer Hold/Watchlist — thesis not fully tested"
            )
            lines.append(
                f"| {portfolio} | {ticker} | {latest_buy} | {days} | {hp} | {ltcg} | {guard} |"
            )

    if not any_recent:
        lines.append("_No positions with a buy within the last 90 days._")
    lines.append("")
    return "\n".join(lines)


def _build_analytical_instructions(
    positions_by_portfolio: Dict,
    today: date,
    cash_by_portfolio: Optional[Dict[str, float]] = None,
) -> str:
    """
    Final user-facing instruction block: maps GrahamPrompt Parts A–K
    to the specific data provided, making the LLM's job precise.
    """
    portfolios_list = ", ".join(f"**{p}**" for p in positions_by_portfolio)
    cash_map   = cash_by_portfolio or {}
    total_cash = sum(cash_map.values())
    cash_lines = "\n".join(
        f"  - {p}: ₹{v:,.0f}" for p, v in cash_map.items() if v > 0
    ) or "  - None recorded"
    return f"""---
## Analysis Instructions (GrahamPrompt Parts A–K)

**Date of analysis:** {today.strftime('%d %b %Y')}
**Portfolios in scope:** {portfolios_list}
**Primary objective:** Achieve ≥20% long-term XIRR; maintain ≥10% XIRR in the near term.

**Available Cash by Portfolio (for Part K — Cash Deployment Plan):**
{cash_lines}
**Total Deployable Cash: ₹{total_cash:,.0f}**

### Your deliverables (strictly follow GrahamPrompt structure):

**Part A – Executive Summary**
Answer: Which portfolio can realistically target 20% XIRR? What is structurally wrong today?

**Part B – Overall Snapshot**
Produce Tables 1–5 (Portfolio-Level Summary, Top Holdings, Sector Allocation,
Stocks Unlikely to Support 20% XIRR, Recent Transactions Requiring Patience).

**Part C – Per-Portfolio Review**
For each portfolio: Diagnosis → Holdings Table → Action Table → Recent Transaction Review
→ Rebalance Plan → Target Portfolio After Rebalance → Conclusion.

Action values must be: **HOLD / TRIM / EXIT / ENTER / ACCUMULATE / WATCHLIST**
Timeline values must be: Immediate | 1–2 weeks | 1 month | 3 months | 6 months |
On decline only | After next quarterly result | After annual report review

**Every TRIM/EXIT must include:**
1. Exact number of shares to sell
2. Whether the trim/exit is too early given holding period (answer Yes/No and why)
3. Which rule triggers it: thesis broken / risk too high / position too large

**Every ENTER/ACCUMULATE must include:**
1. Exact number of shares to buy
2. Target entry price zone
3. Which portfolio weakness it addresses

**Part D – Recommended New Stocks**
If any ENTER actions involve new tickers not currently held, provide Category A/B/C
analysis per GrahamPrompt with write-ups. Fetch or estimate key metrics for NSE-listed
candidates.

**Part E – 5-Year Credibility Review**
Strategic consistency, commitment vs delivery score, and Shenanigans screen for every
current holding using the 5-year P&L and cash flow data provided above.
Classify accounting quality: Clean / Mild Concern / Moderate Concern / Serious Red Flag.

**Part F – Sector Valuation**
Apply sector-appropriate frameworks as specified in the sector notes above.
For Banks/NBFCs do NOT use Graham Number or Current Ratio.

**Part G – Intrinsic Value**
Classify each holding as: Discount to IV / Near Fair Value / Premium to IV.
State whether it deserves capital relative to alternatives for a 20% XIRR target.

**Part H – Rebalance Framework**
Step 1: Propose target allocation ranges per category.
Step 2: Flag all position sizing violations (already pre-computed above — confirm or revise).
Step 3: Full trade table (current % → target % → shares to buy/sell).
Step 4: Phase 1 (Immediate) → Phase 2 (1–3M) → Phase 3 (3–6M) → Phase 4 (Review).

**Part I – Final Action Table** (all portfolios combined)
Columns: Portfolio | Action Type | Symbol | Buy/Sell/Hold | Shares | Trade Value | Timeline | Priority | XIRR Impact

**Part J – Target Portfolio View** (post-rebalance)
Show what each portfolio should look like after all actions are executed.

**Part K – Final Conclusion**
Stocks to Buy | Accumulate on Dips | Trim | Exit | Hold | Recent Purchases to Review Later | Cash Deployment Plan.

---
**Critical rules to enforce:**
- Do not recommend a stock merely because it is cheap.
- Do not recommend buying a high-quality company at any price.
- Downgrade companies with Cash Conversion Ratio <0.8 or active forensic flags.
- Positions with holding period ≤90 days → default to Hold unless thesis is clearly broken.
- Positions approaching LTCG boundary → factor in tax timing in sell recommendations.
- Every large position (>8% portfolio) must be evaluated more deeply than small positions.
- Explicitly state XIRR potential bucket for every stock: >20% / 15–20% / 10–15% / <10% / Unclear.

**Cross-portfolio differentiation (mandatory when ≥2 portfolios exist):**
- Do NOT recommend the same new ENTER in multiple portfolios unless there is a compelling and explicitly stated reason (e.g. the stock is truly exceptional, each portfolio has a completely different mandate, or the combined weight across portfolios is still within concentration limits).
- For each new ENTER recommendation, assign it to the single portfolio where it fits best — based on that portfolio's existing sector gaps, concentration headroom, and cash availability.
- If a sector theme (e.g. IT, Banking) is identified as attractive, split the opportunity: recommend different stocks within that theme across portfolios, not the same stock in all. Example: if IT exposure is needed, put TCS in Portfolio A and a different IT name (e.g. Wipro, Zensar, Persistent) in Portfolio B.
- In Part I (Final Action Table), explicitly flag any ticker that appears as ENTER in more than one portfolio and justify why duplication is warranted despite the added concentration risk.
"""


# ──────────────────────────────────────────────
# NSE Candidate Screener
# ──────────────────────────────────────────────

# Market cap thresholds in INR (confirmed units from live test)
_CAP_LARGE        = 50_000_000_000   # ₹5,000 Cr  — Category A (rebalance)
_CAP_MID          = 20_000_000_000   # ₹2,000 Cr  — Category B (rebalance)
_CAP_SMALL        = 10_000_000_000   # ₹1,000 Cr  — Category C (rebalance)
_CAP_FRESH_MIN    = 80_000_000_000   # ₹8,000 Cr  — fresh portfolio: large + upper mid-cap only

_SCREEN_SIZE_PER_CATEGORY = int(
    os.environ.get("REBALANCE_SCREEN_SIZE_PER_CATEGORY", "30")
)  # candidates fetched per sub-screen before dedup and AI filtering


def _run_screen(query, sort_field: str, size: int) -> List[Dict]:
    """Run yf.screen and return quotes list, empty on failure."""
    try:
        result = yf.screen(query, sortField=sort_field, sortAsc=False, size=size)
        return result.get("quotes") or []
    except Exception as e:
        logger.warning("yf.screen failed: %s", e)
        return []


def _build_sector_aware_queries(cat_id: str, cap_floor: int) -> List[Tuple[str, str, Any, str]]:
    """
    Returns sector-bucketed sub-queries for a given Graham category with
    sector-appropriate P/E thresholds. Banks use P/B; FMCG gets a relaxed
    P/E cap; cyclicals face a tighter P/E cap.

    Returns: [(sector_label, criteria_desc, query, sort_field)]
    """
    from yfinance import EquityQuery

    nse   = EquityQuery("eq", ["exchange", "NSI"])
    cap   = EquityQuery("gt", ["intradaymarketcap", cap_floor])
    fcf_p = EquityQuery("gt", ["leveredfreecashflow.lasttwelvemonths", 0])

    if cat_id == "A":
        return [
            (
                "Financials (Banks / NBFCs)",
                "P/B<3 · ROE>12% · Rev Growth>5% — P/E omitted; use P/B for financials",
                EquityQuery("and", [nse, cap,
                    EquityQuery("lt", ["pricebookratio.quarterly",                  3.0]),
                    EquityQuery("gt", ["pricebookratio.quarterly",                  0.0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.12]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.05]),
                ]),
                "returnonequity.lasttwelvemonths",
            ),
            (
                "Consumer / FMCG",
                "P/E<80 · ROE>20% · Net Margin>12% · Rev Growth>8% · FCF+ve — premium justified",
                EquityQuery("and", [nse, cap, fcf_p,
                    EquityQuery("lt", ["peratio.lasttwelvemonths",                  80]),
                    EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.20]),
                    EquityQuery("gt", ["netincomemargin.lasttwelvemonths",          0.12]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.08]),
                ]),
                "returnonequity.lasttwelvemonths",
            ),
            (
                "Technology / IT Services",
                "P/E<55 · ROE>15% · ROCE>12% · FCF+ve · Rev Growth>8% · LT D/E<1",
                EquityQuery("and", [nse, cap, fcf_p,
                    EquityQuery("lt", ["peratio.lasttwelvemonths",                  55]),
                    EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.15]),
                    EquityQuery("gt", ["returnontotalcapital.lasttwelvemonths",     0.12]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.08]),
                    EquityQuery("lt", ["ltdebtequity.lasttwelvemonths",             1.0]),
                ]),
                "returnonequity.lasttwelvemonths",
            ),
            (
                "Pharma / Manufacturing / General",
                "P/E<45 · ROE>15% · ROCE>12% · Net Margin>10% · Rev Growth>8% · D/E<1 · ICR>3 · FCF+ve",
                EquityQuery("and", [nse, cap, fcf_p,
                    EquityQuery("lt", ["peratio.lasttwelvemonths",                  45]),
                    EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.15]),
                    EquityQuery("gt", ["returnontotalcapital.lasttwelvemonths",     0.12]),
                    EquityQuery("gt", ["netincomemargin.lasttwelvemonths",          0.10]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.08]),
                    EquityQuery("lt", ["ltdebtequity.lasttwelvemonths",             1.0]),
                    EquityQuery("gt", ["ebitinterestexpense.lasttwelvemonths",      3.0]),
                ]),
                "returnonequity.lasttwelvemonths",
            ),
        ]

    if cat_id == "B":
        return [
            (
                "Financial GARP",
                "P/B<2.5 · ROE>18% · Rev Growth>15% · NI Growth>15% — P/E omitted for financials",
                EquityQuery("and", [nse, cap,
                    EquityQuery("lt", ["pricebookratio.quarterly",                  2.5]),
                    EquityQuery("gt", ["pricebookratio.quarterly",                  0.0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.18]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.15]),
                    EquityQuery("gt", ["netincome1yrgrowth.lasttwelvemonths",       0.15]),
                ]),
                "netincome1yrgrowth.lasttwelvemonths",
            ),
            (
                "High-Growth Tech / Consumer GARP",
                "P/E<65 · ROE>15% · Rev Growth>20% · NI Growth>20% · PEG<1.5",
                EquityQuery("and", [nse, cap,
                    EquityQuery("lt", ["peratio.lasttwelvemonths",                  65]),
                    EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.15]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.20]),
                    EquityQuery("gt", ["netincome1yrgrowth.lasttwelvemonths",       0.20]),
                    EquityQuery("lt", ["pegratio_5y",                               1.5]),
                    EquityQuery("gt", ["pegratio_5y",                               0.0]),
                ]),
                "totalrevenues1yrgrowth.lasttwelvemonths",
            ),
            (
                "General GARP",
                "P/E<50 · ROE>15% · Rev Growth>15% · NI Growth>15% · PEG<1.5",
                EquityQuery("and", [nse, cap,
                    EquityQuery("lt", ["peratio.lasttwelvemonths",                  50]),
                    EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                    EquityQuery("gt", ["returnonequity.lasttwelvemonths",           0.15]),
                    EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths",   0.15]),
                    EquityQuery("gt", ["netincome1yrgrowth.lasttwelvemonths",       0.15]),
                    EquityQuery("lt", ["pegratio_5y",                               1.5]),
                    EquityQuery("gt", ["pegratio_5y",                               0.0]),
                ]),
                "totalrevenues1yrgrowth.lasttwelvemonths",
            ),
        ]

    # cat_id == "C"
    return [
        (
            "Cyclicals / Metals Value",
            "P/E<10 · P/B<1.5 · FCF+ve · ICR>3 · Net Margin>5% — tight screen for peak-cycle risk",
            EquityQuery("and", [nse, cap, fcf_p,
                EquityQuery("lt", ["peratio.lasttwelvemonths",                  10]),
                EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                EquityQuery("lt", ["pricebookratio.quarterly",                  1.5]),
                EquityQuery("gt", ["pricebookratio.quarterly",                  0.0]),
                EquityQuery("gt", ["ebitinterestexpense.lasttwelvemonths",      3.0]),
                EquityQuery("gt", ["netincomemargin.lasttwelvemonths",          0.05]),
            ]),
            "returnonequity.lasttwelvemonths",
        ),
        (
            "General Value / Re-Rating",
            "P/E<15 · P/B<2 · FCF+ve · ICR>3 · D/E<1.5 · Net Margin>5%",
            EquityQuery("and", [nse, cap, fcf_p,
                EquityQuery("lt", ["peratio.lasttwelvemonths",                  15]),
                EquityQuery("gt", ["peratio.lasttwelvemonths",                  0]),
                EquityQuery("lt", ["pricebookratio.quarterly",                  2.0]),
                EquityQuery("gt", ["pricebookratio.quarterly",                  0.0]),
                EquityQuery("gt", ["ebitinterestexpense.lasttwelvemonths",      3.0]),
                EquityQuery("lt", ["totaldebtequity.lasttwelvemonths",          1.5]),
                EquityQuery("gt", ["netincomemargin.lasttwelvemonths",          0.05]),
            ]),
            "returnonequity.lasttwelvemonths",
        ),
    ]


def _render_screener_table(
    candidates: List[Tuple[Dict, str, str]],  # (quote_dict, sector_label, criteria)
) -> List[str]:
    """Render a markdown table for screened candidates with separate sector and rule labels."""
    lines = [
        "| Symbol | Company | Sector Bucket | Rule Snapshot | CMP (₹) | Mkt Cap (₹Cr) | P/E | P/B | "
        "Fwd P/E | EPS TTM | Div Yield% | 52W Chg% | Analyst Rating |",
        "|--------|---------|---------------|---------------|---------|--------------|-----|-----|"
        "--------|---------|-----------|----------|----------------|",
    ]
    for c, sector_label, criteria in candidates:
        sym        = c.get("symbol", "")
        name       = (c.get("shortName") or c.get("longName") or "")[:22]
        cmp        = c.get("regularMarketPrice")
        mkt_cap    = c.get("marketCap")
        mkt_cap_cr = round(float(mkt_cap) / 1e7, 0) if mkt_cap not in (None, "") else None
        pe         = round(float(c.get("trailingPE")), 1) if c.get("trailingPE") not in (None, "") else None
        fwd_pe     = round(float(c.get("forwardPE")), 1) if c.get("forwardPE") not in (None, "") else None
        pb         = round(float(c.get("priceToBook")), 2) if c.get("priceToBook") not in (None, "") else None
        eps        = c.get("epsTrailingTwelveMonths")
        div_yld    = _normalize_percent_value(c.get("dividendYield"))
        chg_52w    = _normalize_percent_value(c.get("fiftyTwoWeekChangePercent"))
        rating     = c.get("averageAnalystRating")
        lines.append(
            f"| {sym} | {name} | {sector_label[:22]} | {criteria[:28]} | "
            f"{_fmt_num(cmp, decimals=1, prefix='₹')} | {_fmt_num(mkt_cap_cr, decimals=0, prefix='₹')} | "
            f"{_fmt_text(pe)} | {_fmt_text(pb)} | {_fmt_text(fwd_pe)} | {_fmt_text(eps)} | {_fmt_pct(div_yld, decimals=2)} | "
            f"{_fmt_pct(chg_52w, decimals=1)} | {_fmt_text(rating)} |"
        )
    return lines


def _render_candidate_gate_table(rows: List[Dict[str, Any]], title: str) -> List[str]:
    if not rows:
        return [f"### {title}", "_None_\n"]

    lines = [
        f"### {title}",
        "| Symbol | Company | Status | Key Reason(s) |",
        "|--------|---------|--------|---------------|",
    ]
    for row in rows:
        symbol = row["ticker"]
        company = row["quote"].get("shortName") or row["quote"].get("longName") or symbol
        company = company[:28]
        reasons = "; ".join(row["assessment"]["all_reasons"][:2]) or "Passed forward-risk gate"
        lines.append(
            f"| {symbol} | {company} | {row['assessment']['status']} | {reasons} |"
        )
    lines.append("")
    return lines


def _evaluate_screened_candidates(
    screened_quotes: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], str, List[Dict[str, Any]]]:
    eligible_quotes: Dict[str, Dict[str, Any]] = {}
    eligible_ticker_data: Dict[str, Dict[str, Any]] = {}
    review_rows: List[Dict[str, Any]] = []

    for symbol in sorted(screened_quotes):
        quote = screened_quotes[symbol]
        td = _fetch_ticker_data(symbol)
        event_risks = _check_candidate_events_and_news(symbol)
        assessment = _assess_forward_risk(symbol, quote, td, event_risks)
        review_rows.append({
            "ticker": symbol,
            "quote": quote,
            "ticker_data": td,
            "assessment": assessment,
        })
        if assessment["status"] == "eligible":
            eligible_quotes[symbol] = quote
            eligible_ticker_data[symbol] = td
        logger.info(
            "Candidate gate %s -> %s (%s)",
            symbol,
            assessment["status"],
            "; ".join(assessment["all_reasons"][:2]) or "passed",
        )

    passed = [row for row in review_rows if row["assessment"]["status"] == "eligible"]
    watchlist = [row for row in review_rows if row["assessment"]["status"] == "watchlist"]
    rejected = [row for row in review_rows if row["assessment"]["status"] == "rejected"]

    lines = [
        "## Forward-Risk And Event Gate\n",
        "_Candidates below are filtered by deterministic forward checks before the AI may recommend them. "
        "Rejected names must not be recommended for fresh entry. Watchlist names may be discussed but not selected._\n",
    ]
    lines.extend(_render_candidate_gate_table(passed, "Eligible"))
    lines.extend(_render_candidate_gate_table(watchlist, "Watchlist / Needs Manual Review"))
    lines.extend(_render_candidate_gate_table(rejected, "Rejected Due To Forward Risk"))
    return eligible_quotes, eligible_ticker_data, "\n".join(lines), review_rows


def _screen_nse_candidates(exclude_tickers: set) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Screen NSE candidates across sector-aware sub-queries for Categories A / B / C.
    Excludes tickers already held in the portfolio.
    Returns a markdown section for the AI prompt.
    """
    # Category A uses large-cap floor; B uses mid; C uses small
    category_specs = [
        ("A", "Core Long-Term Compounders", _CAP_LARGE),
        ("B", "GARP Growth Opportunities",  _CAP_MID),
        ("C", "Opportunistic Value / Re-Rating", _CAP_SMALL),
    ]

    seen: set       = set()
    all_quotes: Dict[str, Dict[str, Any]] = {}
    sections: List[str] = [
        "## NSE Candidate Stocks for New Entry (Sector-Aware Screen)\n",
        "_NSE-listed only. Sector-appropriate P/E thresholds: Banks use P/B; "
        f"FMCG P/E<80; Tech P/E<55; Cyclicals P/E<10. Excludes current holdings. "
        f"Wide capture mode: up to {_SCREEN_SIZE_PER_CATEGORY} names per sub-screen before the forward-risk gate and AI ranking._\n",
    ]

    for cat_id, cat_name, cap_floor in category_specs:
        sub_queries = _build_sector_aware_queries(cat_id, cap_floor)
        cat_candidates: List[Tuple[Dict, str, str]] = []

        for sector_label, criteria, query, sort_field in sub_queries:
            quotes = _run_screen(query, sort_field, _SCREEN_SIZE_PER_CATEGORY)
            for q in quotes:
                sym = q.get("symbol", "")
                if sym and sym not in exclude_tickers and sym not in seen:
                    seen.add(sym)
                    cat_candidates.append((q, sector_label, criteria))
                    all_quotes[sym] = q

        sections.append(f"\n### Category {cat_id} — {cat_name}")
        sub_criteria = " | ".join(
            f"**{lbl}**: {crit}"
            for lbl, crit, _, _ in sub_queries
        )
        sections.append(f"_Sub-screens: {sub_criteria}_\n")

        if not cat_candidates:
            sections.append("_No candidates matched after excluding current holdings._\n")
            continue

        sections.extend(_render_screener_table(cat_candidates))
        sections.append("")

    sections.append(
        "> **AI instruction:** For each candidate, validate against GrahamPrompt criteria "
        "(Parts D, E, F, G). Use the **Sector Bucket** and **Rule Snapshot** columns to apply the correct sector valuation "
        "framework — do NOT apply P/E to Financials; use P/B instead. "
        "Recommend ENTER only if the stock is superior to at least one current holding "
        "OR solves a specific portfolio weakness. Suggest exact share count and entry price zone.\n"
    )

    return "\n".join(sections), all_quotes


# ──────────────────────────────────────────────
# Fresh Portfolio Builder
# ──────────────────────────────────────────────

_FRESH_PORTFOLIO_SYSTEM_PROMPT = """
You are a Senior Equity Research Analyst building a brand-new long-term compounding
portfolio from scratch for an investor. Apply the blended Graham/GARP/Quality-Moat
framework strictly.

════════════════════════════════════
OBJECTIVE
════════════════════════════════════
Construct an optimal 10–15 stock portfolio using the cash amount provided.
Target ≥20% long-term XIRR. Every rupee must be allocated — no leftover cash
unless you explicitly justify keeping a reserve.

════════════════════════════════════
STOCK SELECTION CRITERIA
════════════════════════════════════
Category A — Core Long-Term Compounders (target 50–60% of portfolio):
  Durable moat, ROE/ROCE >15%, long reinvestment runway, strong FCF, low leverage,
  clean accounting. Valuation thresholds are SECTOR-SPECIFIC:
    • Banks/NBFCs: P/B <3, ROE >12% — do NOT use P/E for banks
    • Consumer/FMCG: P/E <80 acceptable for durable franchises with ROE >20%
    • Technology: P/E <55, FCF yield must be positive
    • Pharma/Manufacturing/General: P/E <45

Category B — GARP Growth Opportunities (target 25–35%):
  ROE >15%, revenue + earnings growth >15%, PEG <1.5, clean accounting.
    • Financial GARP: P/B <2.5, no P/E filter — use earnings growth + ROE
    • High-growth Tech/Consumer: P/E <65 acceptable if PEG <1.5 and growth >20%
    • General GARP: P/E <50

Category C — Opportunistic Value / Re-Rating (target 10–15% max):
  FCF positive, ICR >3, net margin >5%. Sector-specific:
    • Cyclicals/Metals: P/E <10, P/B <1.5 — tight because peak-cycle earnings inflate P/E
    • General value: P/E <15, P/B <2, D/E <1.5
  Do not let this category dominate — value traps are the biggest risk here.

════════════════════════════════════
POSITION SIZING RULES
════════════════════════════════════
  • Max single stock: 12% of portfolio
  • Min meaningful position: 5%
  • Max cyclical / commodity-linked: 20%
  • Max exposure to <15% XIRR potential stocks: 25%
  • Sector diversification: no single sector >30%

════════════════════════════════════
FORENSIC & QUALITY SCREEN
════════════════════════════════════
Reject any candidate with:
  • Cash Conversion Ratio <0.6 (Serious Red Flag)
  • Net Debt/EBITDA >3x
  • Promoter holding declining sharply (from Screener.in signals)
  • Interest coverage <2x
Downgrade (reduce allocation) for Moderate Concern candidates.

════════════════════════════════════
REQUIRED OUTPUT STRUCTURE
════════════════════════════════════
Section 1 — Portfolio Construction Rationale
  Why these 10–15 stocks? What portfolio weaknesses does the mix avoid?
  Sector balance, category balance, XIRR potential mix.

Section 2 — Stock-by-Stock Analysis
  For each selected stock:
  | Stock | Category | Sector | CMP | Target Alloc % | Target Alloc ₹ | Shares to Buy | Entry Zone | XIRR Potential | Conviction | Key Thesis |
  Followed by: Valuation justification, Forensic screen result, Screener.in signals summary,
  Financial delivery quality (from 5-year trend data provided), Intrinsic Value classification.

Section 3 — Rejected Candidates
  List screened stocks NOT selected and the single reason for rejection each.

Section 4 — Final Portfolio Summary
  | Stock | Sector | Category | Shares | CMP | Allocated ₹ | Portfolio % | XIRR Potential |
  Total deployed vs cash available. Any reserve kept and why.

Section 5 — Execution Plan
  Order to buy (highest conviction first), suggested price limits, what to do if
  a stock gaps up >5% before you can buy.

════════════════════════════════════
OUTPUT QUALITY RULES
════════════════════════════════════
1. Every stock must have exact share count at current CMP.
2. Every allocation must have a specific XIRR potential bucket: >20% / 15–20% / 10–15% / <10% / Unclear.
3. Do not recommend a stock merely because it passes the screen — explain the thesis.
4. Do not recommend a high-quality company at any price — valuation must be justified.
5. Where data is uncertain or missing, say so — do not fabricate.
6. Format entire response as clean Markdown suitable for saving as a .md file.
"""


def _screen_fresh_portfolio_candidates() -> Tuple[str, Dict[str, Dict]]:
    """
    Screen NSE large + upper mid-cap stocks (≥₹8,000 Cr) across sector-aware
    sub-queries for Categories A / B / C.
    Returns (markdown_section, {symbol: quote_dict}).
    """
    category_specs = [
        ("A", "Core Long-Term Compounders"),
        ("B", "GARP Growth Opportunities"),
        ("C", "Opportunistic Value / Re-Rating"),
    ]

    seen: set            = set()
    all_quotes: Dict[str, Dict] = {}
    sections: List[str]  = [
        "## Screened Candidates — Large & Upper Mid-Cap NSE Stocks (≥₹8,000 Cr)\n",
        "_NSE-listed, market cap ≥₹8,000 Cr. Sector-appropriate P/E thresholds applied. "
        f"AI must validate further before recommending. Wide capture mode: up to {_SCREEN_SIZE_PER_CATEGORY} names per sub-screen before the forward-risk gate._\n",
    ]

    for cat_id, cat_name in category_specs:
        sub_queries = _build_sector_aware_queries(cat_id, _CAP_FRESH_MIN)
        cat_candidates: List[Tuple[Dict, str, str]] = []

        for sector_label, criteria, query, sort_field in sub_queries:
            quotes = _run_screen(query, sort_field, _SCREEN_SIZE_PER_CATEGORY)
            for q in quotes:
                sym = q.get("symbol", "")
                if sym and sym not in seen:
                    seen.add(sym)
                    cat_candidates.append((q, sector_label, criteria))
                    all_quotes[sym] = q

        sections.append(f"\n### Category {cat_id} — {cat_name}")
        sub_criteria = " | ".join(
            f"**{lbl}**: {crit}"
            for lbl, crit, _, _ in sub_queries
        )
        sections.append(f"_Sub-screens: {sub_criteria}_\n")

        if not cat_candidates:
            sections.append("_No candidates matched._\n")
            continue

        sections.extend(_render_screener_table(cat_candidates))
        sections.append("")

    return "\n".join(sections), all_quotes


def run_fresh_portfolio_analysis(
    amount: float,
    preferences: Optional[Dict] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Build a fresh portfolio recommendation for the given cash amount.
    preferences keys: risk_appetite, horizon, sector_focus, sector_avoid, stock_count
    Returns (report_path, input_path, usage_msg).
    """
    prefs = preferences or {}
    logger.info(
        "Starting fresh portfolio analysis — amount: ₹%.0f, prefs: %s, model: %s",
        amount, prefs, _REBALANCE_MODEL,
    )

    candidates_section, screened_quotes = _screen_fresh_portfolio_candidates()

    if not screened_quotes:
        return None, None, "📭 No candidates returned from screener — yfinance screen may be unavailable. Try again later."

    eligible_quotes, ticker_data, candidate_gate_section, candidate_rows = _evaluate_screened_candidates(screened_quotes)
    if not eligible_quotes:
        return None, None, "📭 Candidates were screened, but all were rejected by the forward-risk and event gate. Review the watchlist manually or widen the search universe."

    market_data = _build_market_data_section(ticker_data)
    candidate_forward_outlook = _build_candidate_forward_outlook_section(candidate_rows)

    risk       = prefs.get("risk_appetite", "Moderate")
    horizon    = prefs.get("horizon", "Long >5yr")
    focus      = prefs.get("sector_focus", "All sectors")
    avoid      = prefs.get("sector_avoid", "None")
    count_pref = prefs.get("stock_count", "Auto")

    count_instruction = {
        "5–8 (concentrated)":  "Target 5–8 stocks. Prefer high-conviction, concentrated bets.",
        "8–12 (balanced)":     "Target 8–12 stocks. Balance conviction with diversification.",
        "12–15 (diversified)": "Target 12–15 stocks. Prioritise sector diversification.",
        "Auto":                "Choose the optimal number of stocks (typically 8–12).",
    }.get(count_pref, "Choose the optimal number of stocks (typically 8–12).")

    sector_focus_line = (
        f"**Sector Focus:** Overweight {focus} — prioritise candidates from this sector where quality meets criteria."
        if focus != "All sectors" else
        "**Sector Focus:** No preference — allocate across sectors based purely on quality and valuation."
    )
    sector_avoid_line = (
        f"**Sectors to Avoid:** Exclude or severely underweight {avoid} sector stocks."
        if avoid != "None" else
        "**Sectors to Avoid:** None specified."
    )

    user_message = (
        f"## Fresh Portfolio Construction Request\n\n"
        f"**Available Cash:** ₹{amount:,.0f}\n"
        f"**Universe:** NSE large-cap and upper mid-cap stocks (market cap ≥₹8,000 Cr)\n"
        f"**Date:** {date.today().strftime('%d %b %Y')}\n\n"
        f"### Investor Preferences\n"
        f"**Risk Appetite:** {risk}\n"
        f"**Investment Horizon:** {horizon}\n"
        f"{sector_focus_line}\n"
        f"{sector_avoid_line}\n"
        f"**Portfolio Size:** {count_instruction}\n\n"
        f"### Risk Appetite Guidance\n"
        + {
            "Conservative": "- Prefer quality moats with low debt, high FCF, dividend history.\n"
                            "- Avoid cyclicals, turnarounds, high-beta stocks.\n"
                            "- Max single-stock weight: 8%. Min 3 defensive sectors.\n",
            "Moderate":     "- Balance quality compounders with select GARP opportunities.\n"
                            "- Moderate cyclical exposure (≤20% combined) acceptable.\n"
                            "- Max single-stock weight: 12%.\n",
            "Aggressive":   "- Allow high-growth, re-rating, and turnaround candidates.\n"
                            "- Higher single-stock concentration acceptable (up to 15%).\n"
                            "- Sector concentration allowed if thesis is strong.\n",
        }.get(risk, "")
        + f"\n### Horizon Guidance\n"
        + {
            "<2 years":   "- Prefer value unlocking, special situations, or momentum.\n"
                          "- Avoid long gestation businesses or heavy capex cycles.\n",
            "2–5 years":  "- Balance compounders with moderate growth visibility.\n"
                          "- Acceptable to include businesses mid-cycle.\n",
            "Long >5yr":  "- Maximise long-term compounding quality — ROE/ROCE sustainability is paramount.\n"
                          "- Acceptable to pay fair value for exceptional businesses.\n",
        }.get(horizon, "")
        + f"\n{candidates_section}\n\n"
        + f"{candidate_gate_section}\n\n"
        + f"{candidate_forward_outlook}\n\n"
        f"{market_data}\n\n"
        "---\n"
        "Using the screened candidates and their fundamental data above, construct the optimal "
        "portfolio following the required output structure (Sections 1–5). "
        "Strictly respect the investor preferences above. "
        "Allocate exact share counts at current CMP. Justify every inclusion and rejection. "
        "Do not recommend any stock outside the Eligible list. Watchlist names may only be discussed as deferred ideas, not included in the final portfolio."
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    input_md = (
        f"# Fresh Portfolio Analysis — Input Prompt\n"
        f"**Generated:** {datetime.now().strftime('%d %b %Y, %I:%M %p')}  \n"
        f"**Model:** {_REBALANCE_MODEL}  \n"
        f"**Cash Amount:** ₹{amount:,.0f}  \n"
        f"**Risk:** {risk} | **Horizon:** {horizon} | **Focus:** {focus} | **Avoid:** {avoid} | **Count:** {count_pref}  \n\n"
        "---\n\n"
        "## System Prompt\n\n"
        f"{_FRESH_PORTFOLIO_SYSTEM_PROMPT.strip()}\n\n"
        "---\n\n"
        "## User Message\n\n"
        f"{user_message}"
    )
    input_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md",
        prefix=f"fresh_portfolio_input_{timestamp}_",
        delete=False, encoding="utf-8",
    )
    input_tmp.write(input_md)
    input_tmp.flush()
    input_tmp.close()
    logger.info("Fresh portfolio input written to: %s", input_tmp.name)

    try:
        response = openai_model.chat.completions.create(
            model=_REBALANCE_MODEL,
            messages=[
                {"role": "system", "content": _FRESH_PORTFOLIO_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_completion_tokens=16000,
        )
        report  = response.choices[0].message.content.strip()
        usage   = response.usage
        in_tok  = usage.prompt_tokens     if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost_usd = (in_tok / 1_000_000 * _PRICE_INPUT_PER_1M) + (out_tok / 1_000_000 * _PRICE_OUTPUT_PER_1M)
        cost_inr = cost_usd * 84.0
        logger.info(
            "Fresh portfolio done. Tokens — in: %d, out: %d | Cost: $%.4f / ₹%.2f",
            in_tok, out_tok, cost_usd, cost_inr,
        )

        report_md = (
            f"# Fresh Portfolio Recommendations\n"
            f"**Generated:** {datetime.now().strftime('%d %b %Y, %I:%M %p')}  \n"
            f"**Model:** {_REBALANCE_MODEL}  \n"
            f"**Cash Available:** ₹{amount:,.0f}  \n\n"
            "---\n\n"
        ) + report
        report_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md",
            prefix=f"fresh_portfolio_report_{timestamp}_",
            delete=False, encoding="utf-8",
        )
        report_tmp.write(report_md)
        report_tmp.flush()
        report_tmp.close()
        logger.info("Fresh portfolio report written to: %s", report_tmp.name)

        usage_msg = (
            f"📊 Fresh portfolio analysis complete\n"
            f"Model: {_REBALANCE_MODEL}\n"
            f"Input tokens:  {in_tok:,}\n"
            f"Output tokens: {out_tok:,}\n"
            f"Est. cost:     ${cost_usd:.4f} USD  (~₹{cost_inr:.2f})"
        )
        return report_tmp.name, input_tmp.name, usage_msg

    except Exception as e:
        logger.error("Fresh portfolio analysis failed: %s", e, exc_info=True)
        return None, input_tmp.name, f"❌ Fresh portfolio analysis failed: {e}"


# ──────────────────────────────────────────────
# Screener.in qualitative data
# ──────────────────────────────────────────────

_SCREENER_CACHE_DIR  = Path(tempfile.gettempdir()) / "screener_cache"
_SCREENER_CACHE_DAYS = 7


def _fetch_screener_data(ticker: str) -> Dict:
    """
    Fetch Pros, Cons, and About from Screener.in for an NSE/BSE ticker.
    Strips .NS/.BO suffix. Tries consolidated view first, falls back to standalone.
    Results cached for 7 days in /tmp/screener_cache/.
    Returns {"pros": [...], "cons": [...], "about": "...", "symbol": "..."}.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — Screener.in data unavailable. Run: pip install beautifulsoup4")
        return {"pros": [], "cons": [], "about": "", "symbol": ticker}

    symbol = re.sub(r"\.(NS|BO)$", "", ticker.upper())

    _SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _SCREENER_CACHE_DIR / f"{symbol}.json"

    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < _SCREENER_CACHE_DAYS:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    html = None
    for url in [
        f"https://www.screener.in/company/{symbol}/consolidated/",
        f"https://www.screener.in/company/{symbol}/",
    ]:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                html = resp.text
                break
        except Exception as exc:
            logger.debug("Screener.in request failed for %s: %s", url, exc)

    if not html:
        logger.warning("Screener.in: no response for %s", symbol)
        return {"pros": [], "cons": [], "about": "", "symbol": symbol}

    soup = BeautifulSoup(html, "html.parser")

    pros: List[str] = []
    cons: List[str] = []
    analysis = soup.find("section", id="analysis")
    if analysis:
        pros_div = analysis.find("div", class_="pros")
        if pros_div:
            pros = [li.get_text(strip=True) for li in pros_div.find_all("li")]
        cons_div = analysis.find("div", class_="cons")
        if cons_div:
            cons = [li.get_text(strip=True) for li in cons_div.find_all("li")]

    about = ""
    about_tag = soup.find(id="about")
    if about_tag:
        p = about_tag.find("p")
        if p:
            about = p.get_text(strip=True)[:600]

    result = {"pros": pros, "cons": cons, "about": about, "symbol": symbol}
    try:
        cache_file.write_text(json.dumps(result))
    except Exception:
        pass

    logger.info("Screener.in fetched for %s — %d pros, %d cons", symbol, len(pros), len(cons))
    return result


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def prepare_rebalance_analysis(transactions_model) -> Dict[str, Any]:
    """Build rebalance input prompt and write it to disk, without calling the LLM."""
    logger.info("Preparing rebalance analysis — model: %s", _REBALANCE_MODEL)
    transactions = transactions_model.list()
    if not transactions:
        return {"error": "📭 No portfolio transactions found — nothing to analyse."}

    positions_by_portfolio = _compute_positions_by_portfolio(transactions)
    cash_by_portfolio      = _compute_cash_by_portfolio(transactions)

    if not positions_by_portfolio:
        return {"error": "📭 No active (open) positions found."}

    all_tickers = {
        tk
        for tickers in positions_by_portfolio.values()
        for tk in tickers
    }

    logger.info("Fetching market data for %d tickers: %s", len(all_tickers), all_tickers)
    ticker_data: Dict[str, Dict] = {}
    for ticker in sorted(all_tickers):
        ticker_data[ticker] = _fetch_ticker_data(ticker)
        logger.info(
            "Fetched %s — CMP: %s | Revenue CAGR 3Y: %s%% | Cash Conversion: %s | Flags: %d",
            ticker,
            ticker_data[ticker].get("cmp"),
            ticker_data[ticker].get("revenue_cagr_3y_pct"),
            ticker_data[ticker].get("cash_conversion_ratio"),
            len(ticker_data[ticker].get("forensic_flags") or []),
        )

    today = date.today()

    total_current = sum(
        pos["net_shares"] * (ticker_data.get(tk, {}).get("cmp") or 0)
        for tickers in positions_by_portfolio.values()
        for tk, pos in tickers.items()
    )
    total_invested = sum(
        pos["total_invested"] * ticker_data.get(tk, {}).get("fx_rate_to_inr", 1.0)
        for tickers in positions_by_portfolio.values()
        for tk, pos in tickers.items()
    )

    # ── Assemble context sections ──
    positions_table    = _build_positions_table(positions_by_portfolio, ticker_data, today, cash_by_portfolio)
    duplicate_table    = _build_duplicate_exposure_table(positions_by_portfolio, ticker_data, total_current, cash_by_portfolio)
    sizing_violations  = _build_position_sizing_violations(positions_by_portfolio, ticker_data, cash_by_portfolio)
    sector_allocation  = _build_sector_allocation(positions_by_portfolio, ticker_data, total_current, cash_by_portfolio)
    market_data        = _build_market_data_section(ticker_data)
    recent_tx_review   = _build_recent_transaction_review(positions_by_portfolio, today)
    notes_context      = _build_position_notes_section(positions_by_portfolio)
    reconciliation     = _build_reconciliation_checks(transactions)
    instructions       = _build_analytical_instructions(positions_by_portfolio, today, cash_by_portfolio)

    logger.info("Screening NSE candidates for new ENTER actions...")
    candidates_section, screened_quotes = _screen_nse_candidates(all_tickers)
    eligible_quotes, candidate_ticker_data, candidate_gate_section, candidate_rows = _evaluate_screened_candidates(screened_quotes)
    logger.info(
        "NSE screening complete. Eligible: %d / %d",
        len(eligible_quotes),
        len(screened_quotes),
    )
    candidate_market_data = (
        _build_market_data_section(candidate_ticker_data)
        if candidate_ticker_data else
        "## Candidate Deep-Dive\n_No screened candidates passed the forward-risk and event gate._\n"
    )
    forward_outlook = _build_forward_outlook_section(positions_by_portfolio, ticker_data)
    candidate_forward_outlook = _build_candidate_forward_outlook_section(candidate_rows)

    total_cash = sum(cash_by_portfolio.values())
    user_message = (
        f"## Portfolio Rebalance Analysis Request — {today.strftime('%d %b %Y')}\n\n"
        f"**Grand Total Deployed (INR):** ₹{total_invested:,.0f}\n"
        f"**Grand Total Equity Current Value (INR):** ₹{total_current:,.0f}\n"
        f"**Grand Total Cash (INR):** ₹{total_cash:,.0f}\n"
        f"**Grand Total Portfolio Value (Equity + Cash):** ₹{total_current + total_cash:,.0f}\n"
        f"**Overall Unrealised P&L:** ₹{total_current - total_invested:,.0f} "
        f"({((total_current - total_invested) / total_invested * 100) if total_invested else 0:+.1f}%)\n"
        f"**Number of Portfolios:** {len(positions_by_portfolio)}\n"
        f"**Number of Unique Tickers:** {len(all_tickers)}\n\n"
        f"{sector_allocation}\n"
        f"{positions_table}\n"
        f"{duplicate_table}\n"
        f"{sizing_violations}\n"
        f"{recent_tx_review}\n"
        f"{notes_context}\n"
        f"{reconciliation}\n"
        f"{forward_outlook}\n"
        f"{market_data}\n"
        f"{candidates_section}\n"
        f"{candidate_gate_section}\n"
        f"{candidate_forward_outlook}\n"
        f"{candidate_market_data}\n"
        "Only candidates in the Eligible section may be recommended as ENTER actions. "
        "Watchlist names may be mentioned as monitor-only; Rejected names must not be recommended.\n"
        f"{instructions}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    # ── Save input prompt as markdown for independent fact-checking ──
    input_md = (
        f"# Rebalance Analysis — Input Prompt\n"
        f"**Generated:** {datetime.now().strftime('%d %b %Y, %I:%M %p')}  \n"
        f"**Model:** {_REBALANCE_MODEL}  \n\n"
        "---\n\n"
        "## System Prompt (Investment Framework)\n\n"
        f"{_GRAHAM_SYSTEM_PROMPT.strip()}\n\n"
        "---\n\n"
        "## User Message (Portfolio Data + Market Context)\n\n"
        f"{user_message}"
    )
    input_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md",
        prefix=f"rebalance_input_{timestamp}_",
        delete=False, encoding="utf-8",
    )
    input_tmp.write(input_md)
    input_tmp.flush()
    input_tmp.close()
    logger.info("Input prompt written to: %s", input_tmp.name)

    return {
        "input_path": input_tmp.name,
        "input_md": input_md,
        "user_message": user_message,
        "timestamp": timestamp,
        "total_invested": total_invested,
        "total_current": total_current,
    }


def execute_rebalance_analysis(prepared: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Run the LLM for an already prepared rebalance request."""
    user_message = prepared["user_message"]
    timestamp = prepared["timestamp"]
    total_invested = prepared["total_invested"]
    total_current = prepared["total_current"]

    try:
        response = openai_model.chat.completions.create(
            model=_REBALANCE_MODEL,
            messages=[
                {"role": "system", "content": _GRAHAM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_completion_tokens=16000,
        )
        report = response.choices[0].message.content.strip()
        usage  = response.usage
        in_tok  = usage.prompt_tokens     if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost_usd = (in_tok / 1_000_000 * _PRICE_INPUT_PER_1M) + (out_tok / 1_000_000 * _PRICE_OUTPUT_PER_1M)
        cost_inr = cost_usd * 84.0   # rough USD→INR; close enough for billing estimates
        logger.info(
            "Rebalance done. Tokens — in: %d, out: %d | Cost: $%.4f / ₹%.2f",
            in_tok, out_tok, cost_usd, cost_inr,
        )

        # ── Save report as markdown ──
        report_md = (
            f"# Portfolio Rebalance Recommendations\n"
            f"**Generated:** {datetime.now().strftime('%d %b %Y, %I:%M %p')}  \n"
            f"**Model:** {_REBALANCE_MODEL}  \n"
            f"**Total Deployed:** ₹{total_invested:,.0f}  \n"
            f"**Total Current Value:** ₹{total_current:,.0f}  \n\n"
            "---\n\n"
        ) + report
        report_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md",
            prefix=f"rebalance_report_{timestamp}_",
            delete=False, encoding="utf-8",
        )
        report_tmp.write(report_md)
        report_tmp.flush()
        report_tmp.close()
        logger.info("Report written to: %s", report_tmp.name)

        usage_msg = (
            f"📊 Research complete\n"
            f"Model: {_REBALANCE_MODEL}\n"
            f"Input tokens:  {in_tok:,}\n"
            f"Output tokens: {out_tok:,}\n"
            f"Total tokens:  {in_tok + out_tok:,}\n"
            f"Est. cost:     ${cost_usd:.4f} USD  (~₹{cost_inr:.2f})\n"
            f"Pricing used:  ${_PRICE_INPUT_PER_1M}/1M in · ${_PRICE_OUTPUT_PER_1M}/1M out"
        )
        return report_tmp.name, usage_msg

    except Exception as e:
        logger.error("OpenAI rebalance call failed: %s", e, exc_info=True)
        return None, f"❌ Rebalance analysis failed: {e}"


def run_rebalance_analysis(transactions_model) -> str:
    """
    Full Graham-style rebalance analysis across all portfolios.
    Returns (report_path, input_path, usage_msg).
    """
    prepared = prepare_rebalance_analysis(transactions_model)
    if prepared.get("error"):
        return None, None, prepared["error"]
    report_path, usage_msg = execute_rebalance_analysis(prepared)
    return report_path, prepared["input_path"], usage_msg
