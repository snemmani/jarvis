import json
import logging
import re
from datetime import datetime
from typing import Any

import telegram
import wolframalpha
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import ContextTypes

from bujo.analytics.charts import (
    spending_bar_chart,
    spending_category_bar_chart,
    spending_daily_line_chart,
    spending_period_bar_chart,
    spending_pie_chart,
)
from bujo.base import WOLFRAM_APP_ID, expenses_model, llm
from bujo.handlers.utils import send_long
from bujo.managers import expense_manager, mag_manager, portfolio_manager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",
    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",
    "1. Expenses Tool - If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool - If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Portfolio_Tool - If I talk about portfolio transactions (buying or selling stocks/shares, recording a trade, listing my portfolio transactions, depositing or withdrawing cash from a portfolio), use this tool. Also use this tool when the message starts with 'Portfolio transaction:' - that means a screenshot was parsed and you must record it as-is, preserving the Note exactly.",
    "4. Wolfram_Alpha_Tool - If I ask about current events, latest news, mathematical questions, astronomical questions, conversions between units, flight ticket fares, nutrition information of food, stock prices, use this tool.",
    "5. Translation_Tool - If I ask for translation:\n   - And I do not specify source/target languages, assume English to Sanskrit.\n   - If I do specify the languages, translate accordingly.",
    "6. Expense_Analytics_Tool - If I ask for expense charts, spending breakdown, category analysis, daily/weekly/monthly trend, or any analytics/visualisation, use this tool. Pass a JSON string with 'start_date' (YYYY-MM-DD), 'end_date' (YYYY-MM-DD, exclusive upper bound), 'chart_type', optional 'include_terms'/'exclude_terms', and optionally 'request' with my original wording. Supported chart_type values: 'category_pie', 'category_bar', 'daily_bar', 'daily_line', 'weekly_bar', 'monthly_bar'. If I say 'except Home Loan' or 'excluding groceries', put those names in exclude_terms so the chart excludes matching Item/Category rows before plotting. Use the chart type and filters I asked for; do not collapse everything to pie/bar. Example: {\"start_date\": \"2026-05-01\", \"end_date\": \"2026-06-01\", \"chart_type\": \"category_pie\", \"exclude_terms\": [\"Home Loan\"], \"request\": \"May expenses except Home Loan\"}.",
    "MOST IMPORTANT INSTRUCTIONS:",
    "Always call the appropriate tool based on my latest message. You must never answer directly without invoking a tool first.",
    "When sending a response (either to the LLM for summarization or to me), always return it as a string, not as a JSON object.",
    "Always return tool results in markdown format for readability.",
    "Separate individual items in the tool results with new lines.",
    "Use emojis wherever appropriate to make the response more friendly and visually appealing.",
]

_wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)
_main_memory = MemorySaver()

_static_tools = [
    Tool(
        name="Expenses_Interaction",
        func=expense_manager.agent_expenses,
        description="Use this tool to add or list expenses.",
        return_direct=True,
    ),
    Tool(
        name="MAG_interaction",
        func=mag_manager.agent_mag,
        description="Use this tool to manage MAG (calendar).",
        return_direct=True,
    ),
    Tool(
        name="Portfolio_Tool",
        func=portfolio_manager.agent_portfolio,
        description="Use this tool to record or list portfolio transactions (buy/sell stocks, deposit/withdraw cash).",
        return_direct=True,
    ),
    Tool(
        name="Translation_Tool",
        func=lambda x: "This tool must be awaited",
        description="Use this tool to translate from one language to another.",
        coroutine=lambda x: llm.ainvoke([HumanMessage(x)]),
    ),
]


def _loads_tool_json(query: str) -> dict[str, Any]:
    cleaned = str(query or "").replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return {}
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}



def _as_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r",|\band\b|/", str(value), flags=re.IGNORECASE)
    return [term.strip(" '\"`.") for term in raw if term and term.strip(" '\"`.")]


def _terms_from_params(params: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    terms: list[str] = []
    for key in keys:
        terms.extend(_as_terms(params.get(key)))
    return terms


def _extract_excluded_terms(user_text: str, params: dict[str, Any]) -> list[str]:
    terms = _terms_from_params(
        params,
        (
            "exclude_terms",
            "excluded_terms",
            "exclude_items",
            "excluded_items",
            "exclude_categories",
            "excluded_categories",
            "exclude_category",
            "except",
        ),
    )
    if terms:
        return terms

    text = str(user_text or "")
    patterns = [
        r"\b(?:except|excluding|exclude|without|other than)\s+(.+?)(?=$|[?.!]|\s+for\s+(?:the\s+)?(?:chart|graph|breakdown|trend)|\s+as\s+(?:a\s+)?(?:chart|graph))",
        r"\bnot including\s+(.+?)(?=$|[?.!]|\s+for\s+(?:the\s+)?(?:chart|graph|breakdown|trend))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            captured = match.group(1)
            captured = re.sub(r"\b(?:from|in|during)\s+(?:this|last|the)?\s*(?:month|week|year).*", "", captured, flags=re.IGNORECASE)
            return _as_terms(captured)
    return []


def _extract_included_terms(params: dict[str, Any]) -> list[str]:
    return _terms_from_params(
        params,
        (
            "include_terms",
            "included_terms",
            "include_items",
            "included_items",
            "include_categories",
            "included_categories",
            "only_terms",
        ),
    )


def _expense_search_text(expense: dict[str, Any]) -> str:
    return " ".join(
        str(expense.get(field) or "")
        for field in ("Item", "Category", "Type", "Note", "Description")
    ).lower()


def _matches_any(expense: dict[str, Any], terms: list[str]) -> bool:
    haystack = _expense_search_text(expense)
    return any(term.lower() in haystack for term in terms)


def _apply_expense_filters(
    expenses: list[dict[str, Any]],
    params: dict[str, Any],
    user_text: str,
) -> tuple[list[dict[str, Any]], str]:
    include_terms = _extract_included_terms(params)
    exclude_terms = _extract_excluded_terms(user_text, params)

    filtered = expenses
    filter_notes: list[str] = []
    if include_terms:
        before = len(filtered)
        filtered = [expense for expense in filtered if _matches_any(expense, include_terms)]
        filter_notes.append(f"included {', '.join(include_terms)} ({before} -> {len(filtered)})")
    if exclude_terms:
        before = len(filtered)
        filtered = [expense for expense in filtered if not _matches_any(expense, exclude_terms)]
        filter_notes.append(f"excluded {', '.join(exclude_terms)} ({before} -> {len(filtered)})")

    return filtered, "; ".join(filter_notes)

def _choose_chart_type(params: dict[str, Any], user_text: str, tool_query: str) -> str:
    requested = " ".join(
        str(part or "").lower()
        for part in [
            user_text,
            tool_query,
            params.get("request"),
            params.get("chart_type"),
            params.get("group_by"),
            params.get("view"),
        ]
    )
    explicit = str(params.get("chart_type") or "").strip().lower().replace("-", "_").replace(" ", "_")

    # Strong user-language hints win over a generic/default chart_type from the LLM.
    if "line" in requested or "trend line" in requested or "line_graph" in requested:
        return "daily_line"
    if "monthly" in requested or "month_wise" in requested or "month-wise" in requested:
        return "monthly_bar"
    if "weekly" in requested or "week_wise" in requested or "week-wise" in requested:
        return "weekly_bar"
    if "category bar" in requested or "bar by category" in requested or "top categor" in requested:
        return "category_bar"

    aliases = {
        "pie": "category_pie",
        "donut": "category_pie",
        "category": "category_pie",
        "category_pie": "category_pie",
        "breakdown": "category_pie",
        "category_bar": "category_bar",
        "horizontal_bar": "category_bar",
        "bar_by_category": "category_bar",
        "bar": "daily_bar",
        "daily": "daily_bar",
        "daily_bar": "daily_bar",
        "day_bar": "daily_bar",
        "daily_line": "daily_line",
        "line": "daily_line",
        "trend": "daily_line",
        "weekly": "weekly_bar",
        "weekly_bar": "weekly_bar",
        "monthly": "monthly_bar",
        "monthly_bar": "monthly_bar",
    }
    if explicit in aliases:
        return aliases[explicit]
    if "categor" in requested or "breakdown" in requested or "share" in requested:
        return "category_pie"
    return "daily_bar"


async def _make_wolfram_tool(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tool:
    async def query_tool(query: str) -> str:
        logger.info("Querying Wolfram Alpha for: %s", query)
        try:
            response = await _wolfram_client.aquery(query)
            if hasattr(response, "pod"):
                for pod in response["pod"]:
                    subpod = pod["subpod"]
                    if isinstance(subpod, list):
                        for item in subpod:
                            await context.bot.send_photo(
                                chat_id=update.effective_chat.id,
                                photo=item["img"]["@src"],
                                caption=item["@title"],
                            )
                    else:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=subpod["img"]["@src"],
                            caption=pod["@title"],
                        )
            return "Response completed."
        except Exception as e:
            logger.error("Error querying Wolfram Alpha: %s", e)
            return "Wolfram Alpha cannot handle this request, LLM should handle this if it can."

    return Tool(
        name="Wolfram_Alpha_Tool",
        description="Wolfram_Alpha_Tool for complex or realtime queries.",
        func=query_tool,
        coroutine=query_tool,
    )


async def _make_analytics_tool(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> Tool:
    async def analytics_tool(query: str) -> str:
        logger.info("Expense analytics request: %s", query)
        try:
            params = _loads_tool_json(query)
        except Exception:
            params = {"request": str(query or "")}

        start = params.get("start_date")
        end = params.get("end_date")
        chart_type = _choose_chart_type(params, user_text, query)

        filter_parts = []
        if start:
            filter_parts.append(f"(Date,ge,exactDate,{start})")
        if end:
            filter_parts.append(f"(Date,lt,exactDate,{end})")

        expenses = expenses_model.list(
            json.dumps({"filters": filter_parts}) if filter_parts else None
        )
        if not expenses:
            return "No expenses found for that period - nothing to chart."

        original_count = len(expenses)
        expenses, filter_summary = _apply_expense_filters(expenses, params, user_text)
        if not expenses:
            detail = f" after filters ({filter_summary})" if filter_summary else ""
            return f"No expenses found for that period{detail} - nothing to chart."

        grand_total = sum(float(e.get("Amount") or 0) for e in expenses)
        period_label = f"{start} to {end}" if start and end else "All Time"
        filter_line = f"\nFilters: {filter_summary}" if filter_summary else ""

        chart_specs = {
            "category_pie": (
                lambda: spending_pie_chart(expenses, f"Spending Breakdown - {period_label}"),
                "Category breakdown",
            ),
            "category_bar": (
                lambda: spending_category_bar_chart(expenses, f"Category Spending - {period_label}"),
                "Category spending",
            ),
            "daily_bar": (
                lambda: spending_bar_chart(expenses, f"Daily Spending - {period_label}"),
                "Daily spending",
            ),
            "daily_line": (
                lambda: spending_daily_line_chart(expenses, f"Daily Spending Trend - {period_label}"),
                "Daily spending trend",
            ),
            "weekly_bar": (
                lambda: spending_period_bar_chart(expenses, "week", f"Weekly Spending - {period_label}"),
                "Weekly spending",
            ),
            "monthly_bar": (
                lambda: spending_period_bar_chart(expenses, "month", f"Monthly Spending - {period_label}"),
                "Monthly spending",
            ),
        }

        try:
            make_chart, label = chart_specs[chart_type]
            buf = make_chart()
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=buf,
                caption=f"*{label}*\nPeriod: {period_label}{filter_line}\nTotal: Rs.{grand_total:,.0f}",
                parse_mode="markdown",
            )
            return (
                f"Sent {label.lower()} chart ({chart_type}). "
                f"{len(expenses)} of {original_count} transactions plotted, totalling Rs.{grand_total:,.0f} "
                f"for {period_label}."
                + (f" Filters: {filter_summary}." if filter_summary else "")
            )
        except Exception as e:
            logger.error("Chart generation failed: %s", e, exc_info=True)
            return f"Failed to generate chart: {e}"

    return Tool(
        name="Expense_Analytics_Tool",
        description=(
            "Generate visual expense analytics. Supported chart_type values: "
            "category_pie, category_bar, daily_bar, daily_line, weekly_bar, monthly_bar. "
            "Use the chart type requested by the user. Input JSON may include start_date, "
            "end_date, chart_type, group_by, include_terms, exclude_terms, and request. "
            "Apply include/exclude filters to Item/Category before plotting."
        ),
        func=analytics_tool,
        coroutine=analytics_tool,
    )


async def agent_engage(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    tools = _static_tools + [
        await _make_wolfram_tool(update, context),
        await _make_analytics_tool(update, context, text),
    ]

    def _state_modifier(state):
        today = datetime.now().strftime("%Y-%m-%d %A")
        content = "\n".join(SYSTEM_PROMPT) + f"\nToday's date is {today}."
        return [SystemMessage(content=content)] + state["messages"]

    agent = create_react_agent(llm, tools, prompt=_state_modifier, checkpointer=_main_memory)

    logger.info("Chat from user %s: %s", update.effective_user.id, text)
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": f"user_{update.effective_user.id}"}},
        )
        response_text = result["messages"][-1].content
        logger.info("Agent response: %s", response_text)

        if "HERE_IS_IMAGE" in response_text:
            try:
                images_data = response_text.split("\n")
                for image in images_data[1:]:
                    title, image_url = image.split("=>")
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id, photo=image_url, caption=title
                    )
            except Exception as e:
                logger.error("Error sending image: %s", e)
                await update.message.reply_text(f"Error generating image: {e}")
        else:
            await send_long(update.message.reply_text, response_text, parse_mode="markdown")
    except Exception as e:
        logger.error("Error in chat handler: %s", e)
        await update.message.reply_text(f"An error occurred: {e}")
