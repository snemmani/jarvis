import io
from collections import defaultdict
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in Docker / headless
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def _amounts_by_item(expenses: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for exp in expenses:
        item = str(exp.get("Item") or "Misc").strip() or "Misc"
        amount = float(exp.get("Amount") or 0)
        totals[item] += amount
    return dict(totals)


def _amounts_by_date(expenses: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for exp in expenses:
        date = str(exp.get("Date") or "Unknown")[:10]
        amount = float(exp.get("Amount") or 0)
        totals[date] += amount
    return dict(sorted(totals.items()))


# ── chart functions ───────────────────────────────────────────────────────────

def spending_pie_chart(expenses: List[Dict[str, Any]], title: str = "Spending Breakdown") -> io.BytesIO:
    """Pie chart of spending grouped by item.

    Items that account for less than 3 % of the total are merged into 'Others'
    to keep the chart readable.
    """
    totals = _amounts_by_item(expenses)
    if not totals:
        raise ValueError("No expense data available for this period.")

    grand_total = sum(totals.values())
    threshold = grand_total * 0.03

    labels, values, others = [], [], 0.0
    for item, amt in sorted(totals.items(), key=lambda x: -x[1]):
        if amt < threshold:
            others += amt
        else:
            labels.append(item)
            values.append(amt)
    if others > 0:
        labels.append("Others")
        values.append(others)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"Rs.{p * grand_total / 100:,.0f}\n({p:.1f}%)",
        startangle=140,
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(8)

    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold", pad=16)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def spending_bar_chart(expenses: List[Dict[str, Any]], title: str = "Daily Spending") -> io.BytesIO:
    """Bar chart of daily spending totals."""
    totals = _amounts_by_date(expenses)
    if not totals:
        raise ValueError("No expense data available for this period.")

    dates = list(totals.keys())
    amounts = list(totals.values())
    grand_total = sum(amounts)
    max_amt = max(amounts)

    fig_width = max(8, len(dates) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    bars = ax.bar(range(len(dates)), amounts, color="steelblue", edgecolor="white", width=0.7)

    for bar, amt in zip(bars, amounts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_amt * 0.015,
            f"Rs.{amt:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Amount (Rs.)")
    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max_amt * 1.15)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf
