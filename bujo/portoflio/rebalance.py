import logging
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from bujo.base import OPENAI_MODEL, openai_model

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
MANAGEMENT CREDIBILITY (5-year view)
════════════════════════════════════
Assess: strategic consistency, capex delivery, margin promises vs actuals,
debt reduction progress, cash flow improvement, dividend policy consistency,
buyback execution, acquisition discipline, working capital management.
Classify: High Credibility / Mixed Credibility / Low Credibility.
Provide 2–3 evidence examples from multi-year trend data.

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

Part E — 5-Year Credibility & Forensic Review (every current holding)
  Strategic consistency, commitment vs delivery, management sentiment,
  Shenanigans screen, accounting quality classification.

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
                            sell_proceeds, all_buy_lots}}}
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

        if not ticker or shares == 0:
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
        })

        pos = raw[portfolio][ticker]
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
            }

    return result


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

        # ── Core identity & price ──
        d: Dict = {
            "company_name":    info.get("shortName") or info.get("longName") or ticker,
            "sector":          sector,
            "industry":        industry,
            "sector_bucket":   sector_bucket,
            "exchange":        info.get("exchange", "N/A"),
            "market_cap_cr":   round((info.get("marketCap") or 0) / 1e7, 2),
            "cmp":             info.get("currentPrice") or info.get("regularMarketPrice") or 0,
            "52w_high":        info.get("fiftyTwoWeekHigh"),
            "52w_low":         info.get("fiftyTwoWeekLow"),
            "50d_ma":          info.get("fiftyDayAverage"),
            "200d_ma":         info.get("twoHundredDayAverage"),
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
            "book_value":    info.get("bookValue"),
            "eps_ttm":       info.get("trailingEps"),
            "eps_forward":   info.get("forwardEps"),
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
            "dividend_yield":  _pct(info.get("dividendYield")),
            "dividend_rate":   info.get("dividendRate"),
            "payout_ratio":    _pct(info.get("payoutRatio")),
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
            "total_debt_cr":     round(total_debt / 1e7, 2),
            "cash_cr":           round(total_cash / 1e7, 2),
            "net_debt_cr":       round((total_debt - total_cash) / 1e7, 2),
            "total_revenue_cr":  round(total_revenue / 1e7, 2),
            "net_debt_to_ebitda": _safe_div(
                total_debt - total_cash,
                info.get("ebitda") or 0,
            ),
        })

        # ── Cash flow quality (Part E – Shenanigans: OCF vs Net Income) ──
        ocf = info.get("operatingCashflow") or 0
        fcf = info.get("freeCashflow") or 0
        d.update({
            "operating_cashflow_cr": round(ocf / 1e7, 2),
            "free_cash_flow_cr":     round(fcf / 1e7, 2),
            "fcf_yield_pct":         _pct(_safe_div(fcf, market_cap)),
            "capex_cr":              round(((ocf - fcf) if ocf and fcf else 0) / 1e7, 2),
        })

        # ── Ownership & governance (Part E) ──
        d.update({
            "insider_ownership_pct":       _pct(info.get("heldPercentInsiders")),
            "institutional_ownership_pct": _pct(info.get("heldPercentInstitutions")),
            "shares_outstanding_cr":       round((info.get("sharesOutstanding") or 0) / 1e7, 4),
            "float_shares_cr":             round((info.get("floatShares") or 0) / 1e7, 4),
            "short_ratio":                 info.get("shortRatio"),
            "shares_short_pct_float":      _pct(info.get("shortPercentOfFloat")),
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
                            entry[label.lower().replace(" ", "_")] = round(v / 1e7, 2)
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
                            entry[label.lower().replace(" ", "_")] = round(v / 1e7, 2)
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
                            entry[label.lower().replace(" ", "_")] = round(v / 1e7, 2)
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

        return d

    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return {"company_name": ticker, "cmp": 0, "error": str(e)}


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
) -> str:
    """
    Full position table with all GrahamPrompt-required per-position fields:
    portfolio %, holding period classification, LTCG status,
    XIRR proxy, allocation rank.
    """
    lines: List[str] = ["## Current Holdings — Detailed Position Table\n"]

    for portfolio, tickers in positions_by_portfolio.items():
        # Portfolio-level totals for allocation %
        port_current_val = sum(
            pos["net_shares"] * (ticker_data.get(tk, {}).get("cmp") or 0)
            for tk, pos in tickers.items()
        )
        port_invested = sum(pos["total_invested"] for pos in tickers.values())

        lines.append(
            f"### Portfolio: {portfolio}\n"
            f"Invested: ₹{port_invested:,.0f} | Current: ₹{port_current_val:,.0f} | "
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
            cmp     = td.get("cmp") or 0
            avg     = pos["avg_cost"]
            net     = pos["net_shares"]
            cur_val = net * cmp
            inv_val = pos["total_invested"]
            unreal_pct = ((cmp - avg) / avg * 100) if avg > 0 and cmp > 0 else 0.0
            port_pct   = (cur_val / port_current_val * 100) if port_current_val > 0 else 0.0

            # Holding period (from first buy date)
            earliest_buy = min(pos["buy_dates"]) if pos["buy_dates"] else None
            latest_buy   = max(pos["buy_dates"]) if pos["buy_dates"] else None
            holding_days = (today - date.fromisoformat(earliest_buy)).days if earliest_buy else 0
            hp_class     = _classify_holding_period(holding_days)
            ltcg         = _ltcg_status(earliest_buy, today) if earliest_buy else "Unknown"

            # XIRR proxy: simple CAGR from weighted-average buy date to today
            xirr_proxy = None
            if pos["all_buy_lots"] and cmp > 0 and avg > 0:
                # Weighted average purchase date
                total_cost = sum(s * c for _, s, c in pos["all_buy_lots"])
                total_sh   = sum(s for _, s, _ in pos["all_buy_lots"])
                wav_ts = sum(
                    (date.fromisoformat(d) - date(1970, 1, 1)).days * s * c
                    for d, s, c in pos["all_buy_lots"]
                ) / total_cost if total_cost else 0
                wav_date = date(1970, 1, 1) + timedelta(days=wav_ts)
                years_held = (today - wav_date).days / 365.25
                if years_held > 0:
                    xirr_proxy = _cagr(avg, cmp, years_held)

            flags = td.get("forensic_flags") or []
            flags_str = "; ".join(flags) if flags else "None"
            xirr_str = f"{xirr_proxy:+.1f}%" if xirr_proxy is not None else "N/A"

            lines.append(
                f"| {rank} | {ticker} | {td.get('company_name', ticker)[:20]} | "
                f"{td.get('sector_bucket', 'N/A')} | {net:.2f} | "
                f"₹{avg:,.2f} | ₹{cmp:,.2f} | "
                f"₹{inv_val:,.0f} | ₹{cur_val:,.0f} | "
                f"{port_pct:.1f}% | {unreal_pct:+.1f}% | {xirr_str} | "
                f"{holding_days} | {hp_class} | {ltcg} | {flags_str} |"
            )
        lines.append("")

    return "\n".join(lines)


def _build_duplicate_exposure_table(
    positions_by_portfolio: Dict[str, Dict[str, Dict]],
    ticker_data: Dict[str, Dict],
    total_portfolio_value: float,
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
            cross[ticker]["portfolios"].append(portfolio)
            cross[ticker]["total_shares"]   += pos["net_shares"]
            cross[ticker]["total_invested"] += pos["total_invested"]

    dupes = {t: v for t, v in cross.items() if len(v["portfolios"]) > 1}
    if not dupes:
        return "## Cross-Portfolio Duplicate Holdings\nNone — each ticker held in exactly one portfolio.\n"

    lines = [
        "## Cross-Portfolio Duplicate Holdings (GrahamPrompt Level 4)\n",
        "| Ticker | Company | Held In | Total Shares | Total Invested | Total Current | Total Portfolio% | Concentration Risk |",
        "|--------|---------|---------|-------------|---------------|--------------|-------------------|-------------------|",
    ]
    for ticker, info in dupes.items():
        td       = ticker_data.get(ticker, {})
        cmp      = td.get("cmp") or 0
        cur_val  = info["total_shares"] * cmp
        tot_pct  = (cur_val / total_portfolio_value * 100) if total_portfolio_value else 0
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
) -> str:
    """
    Flag positions violating GrahamPrompt Part H Step 2 sizing rules.
    """
    lines = [
        "## Position Sizing Violation Flags (GrahamPrompt Part H Step 2)\n",
        f"Rules: Max single stock {_MAX_SINGLE_STOCK_PCT}% | "
        f"Max low-conviction {_MAX_LOW_CONVICTION_PCT}% | "
        f"Max cyclical {_MAX_CYCLICAL_PCT}%\n",
    ]
    any_violation = False

    for portfolio, tickers in positions_by_portfolio.items():
        port_val = sum(
            pos["net_shares"] * (ticker_data.get(tk, {}).get("cmp") or 0)
            for tk, pos in tickers.items()
        )
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
    }
    for bucket in sorted(sector_val, key=sector_val.get, reverse=True):
        val = sector_val[bucket]
        pct = (val / total_value * 100) if total_value else 0
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

        lines.append(f"### {ticker} — {td.get('company_name', ticker)}")
        lines.append(f"**Sector Bucket:** {td.get('sector_bucket')} | "
                     f"**Exchange:** {td.get('exchange')} | "
                     f"**Industry:** {td.get('industry')}")
        lines.append(f"**Business:** {td.get('business_summary', 'N/A')}")
        lines.append("")

        # Valuation (Part F + G)
        lines.append("**Valuation Multiples:**")
        lines.append(
            f"- P/E TTM: {td.get('pe_ttm')} | P/E Fwd: {td.get('pe_forward')} | "
            f"P/B: {td.get('pb')} | P/S: {td.get('ps')} | EV/EBITDA: {td.get('ev_ebitda')} | PEG: {td.get('peg')}"
        )
        lines.append(
            f"- Book Value/Share: ₹{td.get('book_value')} | EPS TTM: ₹{td.get('eps_ttm')} | EPS Fwd: ₹{td.get('eps_forward')}"
        )
        lines.append(
            f"- Market Cap: ₹{td.get('market_cap_cr')} Cr | FCF Yield: {td.get('fcf_yield_pct')}%"
        )
        lines.append(
            f"- 52W High: ₹{td.get('52w_high')} | 52W Low: ₹{td.get('52w_low')} | "
            f"CMP vs 52W High: {round((td.get('cmp',0) / td.get('52w_high',1) - 1) * 100, 1) if td.get('52w_high') else 'N/A'}% | "
            f"CMP vs 52W Low: {round((td.get('cmp',0) / td.get('52w_low',1) - 1) * 100, 1) if td.get('52w_low') else 'N/A'}%"
        )
        lines.append(
            f"- 50D MA: ₹{td.get('50d_ma')} | 200D MA: ₹{td.get('200d_ma')} | Beta: {td.get('beta')}"
        )

        # Profitability (Part D, F, G)
        lines.append("\n**Profitability & Returns:**")
        lines.append(
            f"- ROE: {td.get('roe')}% | ROA: {td.get('roa')}%"
        )
        lines.append(
            f"- Profit Margin: {td.get('profit_margin')}% | Operating Margin: {td.get('operating_margin')}% | "
            f"Gross Margin: {td.get('gross_margin')}% | EBITDA Margin: {td.get('ebitda_margin')}%"
        )
        lines.append(
            f"- FCF Margin: {td.get('fcf_margin_pct')}% | Interest Coverage: {td.get('interest_coverage')}x"
        )

        # Growth (Part D – GARP)
        lines.append("\n**Growth Profile (GARP Criteria):**")
        lines.append(
            f"- Revenue YoY: {td.get('revenue_growth_yoy')}% | Earnings YoY: {td.get('earnings_growth_yoy')}%"
        )
        lines.append(
            f"- Revenue CAGR 3Y: {td.get('revenue_cagr_3y_pct')}% | 5Y: {td.get('revenue_cagr_5y_pct')}%"
        )
        lines.append(
            f"- Net Income CAGR 3Y: {td.get('net_income_cagr_3y_pct')}% | 5Y: {td.get('net_income_cagr_5y_pct')}%"
        )
        lines.append(
            f"- Price CAGR 5Y: {td.get('price_cagr_5y_pct')}% | 1Y Price Return: {td.get('price_return_1y_pct')}%"
        )
        lines.append(
            f"- PEG Ratio: {td.get('peg')} (GARP target: <1.5 for strong conviction)"
        )

        # Cash flow quality (Part E – Shenanigans)
        lines.append("\n**Cash Flow Quality (Shenanigans Screen):**")
        lines.append(
            f"- Operating CF: ₹{td.get('operating_cashflow_cr')} Cr | FCF: ₹{td.get('free_cash_flow_cr')} Cr | "
            f"Capex: ₹{td.get('capex_cr')} Cr"
        )
        lines.append(
            f"- Cash Conversion Ratio (OCF/Net Income): {td.get('cash_conversion_ratio')} "
            f"(≥0.8 = clean; <0.6 = serious concern)"
        )
        if td.get("annual_cashflows"):
            lines.append("- Annual Cash Flow Trend (₹ Cr):")
            for yr in td["annual_cashflows"]:
                lines.append(
                    f"  {yr['year']}: OCF {yr.get('operating_cash_flow')} Cr | "
                    f"FCF {yr.get('free_cash_flow')} Cr | "
                    f"Capex {yr.get('capital_expenditure')} Cr | "
                    f"Net Debt Chg {yr.get('net_issuance_payments_of_debt')} Cr | "
                    f"Dividends {yr.get('cash_dividends_paid')} Cr"
                )

        # Balance sheet (Part E + sector rules)
        lines.append("\n**Balance Sheet Strength:**")
        lines.append(
            f"- Debt/Equity: {td.get('debt_to_equity')} | Current Ratio: {td.get('current_ratio')} | "
            f"Quick Ratio: {td.get('quick_ratio')}"
        )
        lines.append(
            f"- Total Debt: ₹{td.get('total_debt_cr')} Cr | Cash: ₹{td.get('cash_cr')} Cr | "
            f"Net Debt: ₹{td.get('net_debt_cr')} Cr | Net Debt/EBITDA: {td.get('net_debt_to_ebitda')}x"
        )
        if td.get("annual_balance_sheet"):
            lines.append("- Balance Sheet Trend (₹ Cr):")
            for yr in td["annual_balance_sheet"]:
                lines.append(
                    f"  {yr['year']}: Receivables {yr.get('accounts_receivable')} | "
                    f"Inventory {yr.get('inventory')} | "
                    f"Equity {yr.get('stockholders_equity')} | "
                    f"LT Debt {yr.get('long_term_debt')}"
                )

        # Annual P&L (Part E – 5-year trend)
        lines.append("\n**Annual P&L Trend (₹ Cr) — 5-Year History:**")
        if td.get("annual_financials"):
            for yr in td["annual_financials"]:
                lines.append(
                    f"  {yr['year']}: Revenue {yr.get('total_revenue')} | "
                    f"Gross Profit {yr.get('gross_profit')} | "
                    f"EBIT {yr.get('ebit')} | "
                    f"Net Income {yr.get('net_income')}"
                )
        else:
            lines.append("  _Not available_")

        # Dividend (income / value anchor role)
        lines.append("\n**Dividend & Shareholder Returns:**")
        lines.append(
            f"- Dividend Yield: {td.get('dividend_yield')}% | Rate: ₹{td.get('dividend_rate')} | "
            f"Payout Ratio: {td.get('payout_ratio')}%"
        )

        # Ownership & governance (Part E)
        lines.append("\n**Ownership & Governance:**")
        lines.append(
            f"- Insider: {td.get('insider_ownership_pct')}% | Institutional: {td.get('institutional_ownership_pct')}%"
        )
        lines.append(
            f"- Shares Outstanding: {td.get('shares_outstanding_cr')} Cr | "
            f"Short Ratio: {td.get('short_ratio')} | Short % Float: {td.get('shares_short_pct_float')}%"
        )

        # Forensic flags (Part E)
        flags = td.get("forensic_flags") or []
        lines.append("\n**Forensic Accounting Signals (Part E Screen):**")
        if flags:
            for f in flags:
                lines.append(f"  {f}")
        else:
            lines.append("  ✅ No automated forensic flags detected")

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


def _build_analytical_instructions(positions_by_portfolio: Dict, today: date) -> str:
    """
    Final user-facing instruction block: maps GrahamPrompt Parts A–K
    to the specific data provided, making the LLM's job precise.
    """
    portfolios_list = ", ".join(f"**{p}**" for p in positions_by_portfolio)
    return f"""---
## Analysis Instructions (GrahamPrompt Parts A–K)

**Date of analysis:** {today.strftime('%d %b %Y')}
**Portfolios in scope:** {portfolios_list}
**Primary objective:** Achieve ≥20% long-term XIRR; maintain ≥10% XIRR in the near term.

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
"""


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run_rebalance_analysis(transactions_model) -> str:
    """
    Full Graham-style rebalance analysis across all portfolios.
    Returns a formatted report string for Telegram delivery.
    """
    logger.info("Starting rebalance analysis — model: %s", _REBALANCE_MODEL)

    transactions = transactions_model.list()
    if not transactions:
        return "📭 No portfolio transactions found — nothing to analyse."

    positions_by_portfolio = _compute_positions_by_portfolio(transactions)
    if not positions_by_portfolio:
        return "📭 No active (open) positions found."

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
        pos["total_invested"]
        for tickers in positions_by_portfolio.values()
        for pos in tickers.values()
    )

    # ── Assemble context sections ──
    positions_table    = _build_positions_table(positions_by_portfolio, ticker_data, today)
    duplicate_table    = _build_duplicate_exposure_table(positions_by_portfolio, ticker_data, total_current)
    sizing_violations  = _build_position_sizing_violations(positions_by_portfolio, ticker_data)
    sector_allocation  = _build_sector_allocation(positions_by_portfolio, ticker_data, total_current)
    market_data        = _build_market_data_section(ticker_data)
    recent_tx_review   = _build_recent_transaction_review(positions_by_portfolio, today)
    instructions       = _build_analytical_instructions(positions_by_portfolio, today)

    user_message = (
        f"## Portfolio Rebalance Analysis Request — {today.strftime('%d %b %Y')}\n\n"
        f"**Grand Total Deployed (INR):** ₹{total_invested:,.0f}\n"
        f"**Grand Total Current Value (INR):** ₹{total_current:,.0f}\n"
        f"**Overall Unrealised P&L:** ₹{total_current - total_invested:,.0f} "
        f"({((total_current - total_invested) / total_invested * 100) if total_invested else 0:+.1f}%)\n"
        f"**Number of Portfolios:** {len(positions_by_portfolio)}\n"
        f"**Number of Unique Tickers:** {len(all_tickers)}\n\n"
        f"{sector_allocation}\n"
        f"{positions_table}\n"
        f"{duplicate_table}\n"
        f"{sizing_violations}\n"
        f"{recent_tx_review}\n"
        f"{market_data}\n"
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
        return report_tmp.name, input_tmp.name, usage_msg

    except Exception as e:
        logger.error("OpenAI rebalance call failed: %s", e, exc_info=True)
        # Still return the input file so the user can inspect what was sent
        return None, input_tmp.name, f"❌ Rebalance analysis failed: {e}"
