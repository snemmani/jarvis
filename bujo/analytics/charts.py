import io
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _amounts_by_item(expenses: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for exp in expenses:
        item = str(exp.get("Item") or exp.get("Category") or "Misc").strip() or "Misc"
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


def _amounts_by_period(expenses: List[Dict[str, Any]], period: str) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for exp in expenses:
        raw_date = str(exp.get("Date") or "")[:10]
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            label = raw_date or "Unknown"
        else:
            if period == "week":
                year, week, _ = dt.isocalendar()
                label = f"{year}-W{week:02d}"
            else:
                label = dt.strftime("%Y-%m")
        totals[label] += float(exp.get("Amount") or 0)
    return dict(sorted(totals.items()))


def _save(fig) -> io.BytesIO:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def spending_pie_chart(expenses: List[Dict[str, Any]], title: str = "Spending Breakdown") -> io.BytesIO:
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
    _, _, autotexts = ax.pie(
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
    return _save(fig)


def spending_category_bar_chart(expenses: List[Dict[str, Any]], title: str = "Category Spending") -> io.BytesIO:
    totals = _amounts_by_item(expenses)
    if not totals:
        raise ValueError("No expense data available for this period.")

    rows = sorted(totals.items(), key=lambda x: x[1])
    labels = [r[0] for r in rows]
    amounts = [r[1] for r in rows]
    grand_total = sum(amounts)
    fig_height = max(5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    bars = ax.barh(range(len(labels)), amounts, color="#4C78A8")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Amount (Rs.)")
    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    max_amt = max(amounts) if amounts else 0
    for bar, amt in zip(bars, amounts):
        ax.text(
            bar.get_width() + max_amt * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"Rs.{amt:,.0f}",
            va="center",
            fontsize=8,
        )
    plt.tight_layout()
    return _save(fig)


def spending_bar_chart(expenses: List[Dict[str, Any]], title: str = "Daily Spending") -> io.BytesIO:
    totals = _amounts_by_date(expenses)
    if not totals:
        raise ValueError("No expense data available for this period.")

    dates = list(totals.keys())
    amounts = list(totals.values())
    grand_total = sum(amounts)
    max_amt = max(amounts)

    fig_width = max(8, len(dates) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(range(len(dates)), amounts, color="#4C78A8", edgecolor="white", width=0.7)

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
    ax.set_xticklabels([d[5:] if len(d) >= 10 else d for d in dates], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Amount (Rs.)")
    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max_amt * 1.15 if max_amt else 1)
    plt.tight_layout()
    return _save(fig)


def spending_daily_line_chart(expenses: List[Dict[str, Any]], title: str = "Daily Spending Trend") -> io.BytesIO:
    totals = _amounts_by_date(expenses)
    if not totals:
        raise ValueError("No expense data available for this period.")

    dates = list(totals.keys())
    amounts = list(totals.values())
    grand_total = sum(amounts)
    fig_width = max(8, len(dates) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    ax.plot(range(len(dates)), amounts, marker="o", color="#2F6F73", linewidth=2)
    ax.fill_between(range(len(dates)), amounts, color="#2F6F73", alpha=0.15)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] if len(d) >= 10 else d for d in dates], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Amount (Rs.)")
    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig)


def spending_period_bar_chart(expenses: List[Dict[str, Any]], period: str, title: str = "Period Spending") -> io.BytesIO:
    totals = _amounts_by_period(expenses, period)
    if not totals:
        raise ValueError("No expense data available for this period.")

    labels = list(totals.keys())
    amounts = list(totals.values())
    grand_total = sum(amounts)
    max_amt = max(amounts)
    fig_width = max(8, len(labels) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    bars = ax.bar(range(len(labels)), amounts, color="#F58518", edgecolor="white", width=0.7)
    for bar, amt in zip(bars, amounts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_amt * 0.015,
            f"Rs.{amt:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Amount (Rs.)")
    ax.set_title(f"{title}\nTotal: Rs.{grand_total:,.0f}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max_amt * 1.15 if max_amt else 1)
    plt.tight_layout()
    return _save(fig)
