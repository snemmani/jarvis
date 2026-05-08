import asyncio
import logging
import os
from urllib.parse import quote, unquote

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

from bujo.base import check_authorization, portfolio_transactions_model, CHAT_ID
from bujo.managers import portfolio_manager
from bujo.portoflio.alerts import run_portfolio_alerts
from bujo.portoflio.rebalance import (
    prepare_rebalance_analysis,
    execute_rebalance_analysis,
    run_rebalance_analysis,
    run_fresh_portfolio_analysis,
)
from bujo.handlers.utils import send_long

logger = logging.getLogger(__name__)

BP_AMOUNT, BP_RISK, BP_HORIZON, BP_SECTOR_FOCUS, BP_SECTOR_AVOID, BP_STOCK_COUNT, BP_CONFIRM = range(7)

_BP_SECTORS = ["IT / Tech", "Banking / Finance", "Pharma", "FMCG / Consumer", "Manufacturing", "Energy"]
_PORTFOLIO_DASHBOARD_PREFIX = "pdash_"
_REBALANCE_APPROVAL_PREFIX = "rbal_"
_rebalance_pending: dict[int, dict] = {}


@check_authorization
async def get_cmp_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = portfolio_manager.update_cmp()
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="markdown")
    except Exception as e:
        logger.error("Error in get_cmp_today: %s", e)


@check_authorization
async def portfolio_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("portfolioAlerts from user %s", update.effective_user.id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        msg = await asyncio.to_thread(run_portfolio_alerts, portfolio_transactions_model)
        if msg:
            await send_long(update.message.reply_text, msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("✅ No alerts — all positions look clean.")
    except Exception as e:
        logger.error("Error in portfolio_alerts: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error running alerts: {e}")


@check_authorization
async def rebalance_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("rebalanceRecommendations from user %s", update.effective_user.id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        prepared = await asyncio.to_thread(
            prepare_rebalance_analysis, portfolio_transactions_model
        )
        if prepared.get("error"):
            await update.message.reply_text(prepared["error"])
            return

        _rebalance_pending[update.effective_user.id] = prepared
        input_path = prepared["input_path"]
        if input_path and os.path.exists(input_path):
            with open(input_path, "rb") as f:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=f,
                    filename=os.path.basename(input_path),
                    caption="📋 Rebalance input prompt — review this before sending to the model",
                )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Forward To LLM", callback_data=f"{_REBALANCE_APPROVAL_PREFIX}yes"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"{_REBALANCE_APPROVAL_PREFIX}no"),
        ]])
        await update.message.reply_text(
            "Prompt generated. Should I forward it to the model and fetch the rebalance report?",
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.error("Error in rebalanceRecommendations: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error running rebalance analysis: {e}")


async def rebalance_approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    prepared = _rebalance_pending.get(user_id)

    if query.data == f"{_REBALANCE_APPROVAL_PREFIX}no":
        if prepared:
            input_path = prepared.get("input_path")
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            _rebalance_pending.pop(user_id, None)
        await query.edit_message_text("❌ Rebalance request cancelled.")
        return

    if not prepared:
        await query.edit_message_text("❌ No pending rebalance prompt found. Run `/rebalanceRecommendations` again.", parse_mode="Markdown")
        return

    await query.edit_message_text("⏳ Forwarding prompt to the model and generating the rebalance report…")
    try:
        report_path, usage_msg = await asyncio.to_thread(execute_rebalance_analysis, prepared)
        input_path = prepared.get("input_path")
        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as f:
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=f,
                    filename=os.path.basename(report_path),
                    caption="📄 Rebalance report — open in any Markdown viewer",
                )
            os.remove(report_path)
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        _rebalance_pending.pop(user_id, None)
        await query.message.reply_text(usage_msg)
    except Exception as e:
        logger.error("Error in rebalance_approval_callback: %s", e, exc_info=True)
        await query.message.reply_text(f"❌ Error generating rebalance report: {e}")


def _fmt_inr(value: float, decimals: int = 0) -> str:
    return f"₹{value:,.{decimals}f}"


def _pl_emoji(value: float) -> str:
    return "🟢" if value >= 0 else "🔴"


def _encode_portfolio(name: str) -> str:
    return quote(name, safe="")


def _decode_portfolio(name: str) -> str:
    return unquote(name)


def _dashboard_keyboard(dashboard: dict, portfolio_name: str | None = None) -> InlineKeyboardMarkup:
    portfolios = dashboard.get("portfolios") or {}
    rows = [[InlineKeyboardButton("Overview", callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}home")]]
    portfolio_buttons = [
        InlineKeyboardButton(name, callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}portfolio:{_encode_portfolio(name)}")
        for name in sorted(portfolios)
    ]
    for idx in range(0, len(portfolio_buttons), 2):
        rows.append(portfolio_buttons[idx:idx + 2])

    if portfolio_name and portfolio_name in portfolios:
        rows.append([
            InlineKeyboardButton("Holdings", callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}view:holdings:{_encode_portfolio(portfolio_name)}"),
            InlineKeyboardButton("Cash", callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}view:cash:{_encode_portfolio(portfolio_name)}"),
        ])
        rows.append([
            InlineKeyboardButton("Risk", callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}view:risk:{_encode_portfolio(portfolio_name)}"),
            InlineKeyboardButton("Rebalance", callback_data=f"{_PORTFOLIO_DASHBOARD_PREFIX}view:rebalance:{_encode_portfolio(portfolio_name)}"),
        ])

    return InlineKeyboardMarkup(rows)


def _portfolio_risk_flags(portfolio: dict, total_portfolio_value: float) -> list[str]:
    flags: list[str] = []
    holdings = portfolio.get("holdings") or []
    for item in holdings:
        weight = (item["current_inr"] / total_portfolio_value * 100) if total_portfolio_value else 0.0
        if weight > 25:
            flags.append(f"{item['ticker']} is {weight:.1f}% of {portfolio['name']}")
    if portfolio.get("cash_inr", 0.0) < 0:
        flags.append(f"{portfolio['name']} has negative cash: {_fmt_inr(portfolio['cash_inr'])}")
    missing_cmp = [item for item in holdings if item["cmp_inr"] <= 0]
    if missing_cmp:
        flags.append(f"{len(missing_cmp)} holding(s) in {portfolio['name']} missing CMP")
    if not flags:
        flags.append(f"No immediate risk flags for {portfolio['name']}.")
    return flags


def _render_overview(dashboard: dict) -> str:
    totals = dashboard.get("totals") or {}
    portfolios = dashboard.get("portfolios") or {}
    holdings = dashboard.get("holdings") or []
    risk_flags = dashboard.get("risk_flags") or []
    as_of = dashboard.get("as_of")
    timestamp = as_of.strftime("%d %b %Y, %I:%M %p") if as_of else "Unknown"

    if not portfolios:
        return "📭 No portfolio transactions found."

    lines = [
        "📊 *Portfolio Dashboard — Overview*\n"
        f"🕐 {timestamp}",
        "",
        f"Total Value: {_fmt_inr(totals.get('total_value_inr', 0.0))}",
        f"Equity Value: {_fmt_inr(totals.get('current_inr', 0.0))}",
        f"Cash: {_fmt_inr(totals.get('cash_inr', 0.0))}",
        f"Unrealised P&L: {_pl_emoji(totals.get('unrealised_pl', 0.0))} "
        f"{_fmt_inr(totals.get('unrealised_pl', 0.0))} ({totals.get('unrealised_pct', 0.0):+.1f}%)",
        f"Portfolios: {totals.get('portfolio_count', 0)}",
        "",
        "*Portfolios*",
    ]
    for item in sorted(portfolios.values(), key=lambda row: row["total_value_inr"], reverse=True):
        lines.append(
            f"- *{item['name']}*: {_fmt_inr(item['total_value_inr'])} "
            f"(Cash {_fmt_inr(item['cash_inr'])}, {_pl_emoji(item['unrealised_pl'])} P&L {_fmt_inr(item['unrealised_pl'])})"
        )
    if holdings:
        lines.extend([
            "",
            f"Top Holding: {holdings[0]['ticker']} at {holdings[0]['weight_pct']:.1f}%",
            f"Risk Flag: {risk_flags[0]}",
            "",
            "Select a portfolio below for scoped views.",
        ])
    return "\n".join(lines)


def _render_portfolio_dashboard_view(view: str, portfolio: dict, dashboard: dict) -> str:
    as_of = dashboard.get("as_of")
    timestamp = as_of.strftime("%d %b %Y, %I:%M %p") if as_of else "Unknown"
    holdings = portfolio.get("holdings") or []
    total_value = portfolio.get("total_value_inr", 0.0)
    risk_flags = _portfolio_risk_flags(portfolio, total_value)

    if view == "holdings":
        lines = [
            f"📈 *{portfolio['name']} — Holdings*",
            f"🕐 {timestamp}",
            "",
        ]
        for idx, item in enumerate(holdings[:10], start=1):
            weight = (item["current_inr"] / total_value * 100) if total_value else 0.0
            lines.append(
                f"{idx}. *{item['ticker']}*\n"
                f"Value: {_fmt_inr(item['current_inr'])} | Weight: {weight:.1f}%\n"
                f"{_pl_emoji(item['unrealised_pl'])} P&L {_fmt_inr(item['unrealised_pl'])} ({item['unrealised_pct']:+.1f}%)"
            )
        return "\n".join(lines)

    if view == "cash":
        return (
            f"💵 *{portfolio['name']} — Cash*\n"
            f"🕐 {timestamp}\n\n"
            f"Cash Balance: {_fmt_inr(portfolio.get('cash_inr', 0.0))}\n"
            f"Equity Value: {_fmt_inr(portfolio.get('current_inr', 0.0))}\n"
            f"Total Portfolio Value: {_fmt_inr(total_value)}"
        )

    if view == "risk":
        lines = [
            f"⚠️ *{portfolio['name']} — Risk*",
            f"🕐 {timestamp}",
            "",
            *[f"- {flag}" for flag in risk_flags],
        ]
        if holdings:
            lines.extend(["", "Top exposure:"])
            for item in holdings[:5]:
                weight = (item["current_inr"] / total_value * 100) if total_value else 0.0
                lines.append(f"- {item['ticker']}: {weight:.1f}% of {portfolio['name']}")
        return "\n".join(lines)

    if view == "rebalance":
        return (
            f"🧭 *{portfolio['name']} — Rebalance*\n"
            f"🕐 {timestamp}\n\n"
            f"Current value: {_fmt_inr(total_value)}\n"
            f"Cash available: {_fmt_inr(portfolio.get('cash_inr', 0.0))}\n"
            f"Open holdings: {len(holdings)}\n\n"
            "Use `/rebalanceRecommendations` for the full cross-portfolio rebalance report."
        )

    lines = [
        f"💼 *{portfolio['name']}*",
        f"🕐 {timestamp}",
        "",
        f"Total Value: {_fmt_inr(total_value)}",
        f"Equity Value: {_fmt_inr(portfolio.get('current_inr', 0.0))}",
        f"Cash: {_fmt_inr(portfolio.get('cash_inr', 0.0))}",
        f"Unrealised P&L: {_pl_emoji(portfolio.get('unrealised_pl', 0.0))} {_fmt_inr(portfolio.get('unrealised_pl', 0.0))}",
        f"Holdings: {len(holdings)}",
    ]
    if holdings:
        top = holdings[0]
        weight = (top["current_inr"] / total_value * 100) if total_value else 0.0
        lines.extend([
            "",
            f"Top Holding: {top['ticker']} at {weight:.1f}%",
            f"Risk Flag: {risk_flags[0]}",
        ])
    return "\n".join(lines)


@check_authorization
async def portfolio_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("portfolioDashboard from user %s", update.effective_user.id)
    await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
    try:
        await asyncio.to_thread(portfolio_manager.update_cmp)
        dashboard = await asyncio.to_thread(portfolio_manager.get_dashboard_data)
        text = _render_overview(dashboard)
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=_dashboard_keyboard(dashboard))
    except Exception as e:
        logger.error("Error in portfolio_dashboard: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Error generating dashboard: {e}")


async def portfolio_dashboard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        dashboard = await asyncio.to_thread(portfolio_manager.get_dashboard_data)
        action = query.data.removeprefix(_PORTFOLIO_DASHBOARD_PREFIX)
        portfolio_name = None

        if action == "home":
            text = _render_overview(dashboard)
        elif action.startswith("portfolio:"):
            portfolio_name = _decode_portfolio(action.split(":", 1)[1])
            portfolio = (dashboard.get("portfolios") or {}).get(portfolio_name)
            if not portfolio:
                await query.edit_message_text("❌ Portfolio not found. Refresh with `/portfolioDashboard`.", parse_mode="Markdown")
                return
            text = _render_portfolio_dashboard_view("summary", portfolio, dashboard)
        elif action.startswith("view:"):
            _, view, encoded_name = action.split(":", 2)
            portfolio_name = _decode_portfolio(encoded_name)
            portfolio = (dashboard.get("portfolios") or {}).get(portfolio_name)
            if not portfolio:
                await query.edit_message_text("❌ Portfolio not found. Refresh with `/portfolioDashboard`.", parse_mode="Markdown")
                return
            text = _render_portfolio_dashboard_view(view, portfolio, dashboard)
        else:
            text = _render_overview(dashboard)

        await query.edit_message_text(
            text,
            parse_mode="Markdown",
            reply_markup=_dashboard_keyboard(dashboard, portfolio_name=portfolio_name),
        )
    except Exception as e:
        logger.error("Error in portfolio_dashboard_callback: %s", e, exc_info=True)
        await query.message.reply_text(f"❌ Error updating dashboard: {e}")


# ── Build Portfolio conversation ──────────────────────────────────────────────

@check_authorization
async def bp_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("buildPortfolio started by user %s", update.effective_user.id)
    context.user_data.clear()
    args = context.args
    if args:
        try:
            amount = float(args[0].replace(",", "").replace("₹", ""))
            context.user_data["bp_amount"] = amount
            return await _bp_ask_risk(update, context)
        except ValueError:
            pass
    await update.message.reply_text(
        "💼 *Build a Fresh Portfolio*\n\nStep 1/6 — How much do you want to invest?\n"
        "Send the amount in ₹ (e.g. `500000` or `5,00,000`)",
        parse_mode="Markdown",
    )
    return BP_AMOUNT


async def bp_got_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().replace(",", "").replace("₹", "")
    try:
        amount = float(text)
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Please send a valid positive number, e.g. `500000`.")
        return BP_AMOUNT
    context.user_data["bp_amount"] = amount
    return await _bp_ask_risk(update, context)


async def _bp_ask_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🛡️ Conservative", callback_data="bp_risk_Conservative"),
        InlineKeyboardButton("⚖️ Moderate",     callback_data="bp_risk_Moderate"),
        InlineKeyboardButton("🚀 Aggressive",   callback_data="bp_risk_Aggressive"),
    ]])
    amount = context.user_data["bp_amount"]
    msg = f"✅ Amount: ₹{amount:,.0f}\n\nStep 2/6 — What's your risk appetite?"
    if update.callback_query:
        await update.callback_query.message.reply_text(msg, reply_markup=keyboard)
    else:
        await update.message.reply_text(msg, reply_markup=keyboard)
    return BP_RISK


async def bp_got_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["bp_risk"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📅 Short  <2yr",   callback_data="bp_horizon_<2 years"),
        InlineKeyboardButton("📆 Medium 2–5yr", callback_data="bp_horizon_2–5 years"),
        InlineKeyboardButton("🗓️ Long  >5yr",   callback_data="bp_horizon_Long >5yr"),
    ]])
    await query.message.reply_text(
        f"✅ Risk: {context.user_data['bp_risk']}\n\nStep 3/6 — Investment horizon?",
        reply_markup=keyboard,
    )
    return BP_HORIZON


async def bp_got_horizon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["bp_horizon"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    buttons = [[InlineKeyboardButton("🌐 All sectors", callback_data="bp_focus_All sectors")]]
    buttons += [[InlineKeyboardButton(s, callback_data=f"bp_focus_{s}")] for s in _BP_SECTORS]
    await query.message.reply_text(
        f"✅ Horizon: {context.user_data['bp_horizon']}\n\nStep 4/6 — Any sector you want to *focus* on?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return BP_SECTOR_FOCUS


async def bp_got_focus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["bp_focus"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    buttons = [[InlineKeyboardButton("🚫 None", callback_data="bp_avoid_None")]]
    buttons += [[InlineKeyboardButton(s, callback_data=f"bp_avoid_{s}")] for s in _BP_SECTORS]
    await query.message.reply_text(
        f"✅ Focus: {context.user_data['bp_focus']}\n\nStep 5/6 — Any sector to *avoid*?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return BP_SECTOR_AVOID


async def bp_got_avoid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["bp_avoid"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("5–8 (concentrated)",  callback_data="bp_count_5–8 (concentrated)"),
        InlineKeyboardButton("8–12 (balanced)",     callback_data="bp_count_8–12 (balanced)"),
    ], [
        InlineKeyboardButton("12–15 (diversified)", callback_data="bp_count_12–15 (diversified)"),
        InlineKeyboardButton("🤖 Auto",             callback_data="bp_count_Auto"),
    ]])
    await query.message.reply_text(
        f"✅ Avoid: {context.user_data['bp_avoid']}\n\nStep 6/6 — How many stocks?",
        reply_markup=keyboard,
    )
    return BP_STOCK_COUNT


async def bp_got_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["bp_count"] = query.data.split("_", 2)[-1]
    await query.message.delete()
    d = context.user_data
    summary = (
        f"📋 *Portfolio Build Summary*\n\n"
        f"💰 Amount:  ₹{d['bp_amount']:,.0f}\n"
        f"⚖️ Risk:    {d['bp_risk']}\n"
        f"🗓️ Horizon: {d['bp_horizon']}\n"
        f"🎯 Focus:   {d['bp_focus']}\n"
        f"🚫 Avoid:   {d['bp_avoid']}\n"
        f"📊 Stocks:  {d['bp_count']}\n\n"
        "Shall I build this portfolio? (takes ~2–3 minutes)"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Build it", callback_data="bp_confirm_yes"),
        InlineKeyboardButton("❌ Cancel",   callback_data="bp_confirm_no"),
    ]])
    await query.message.reply_text(summary, parse_mode="Markdown", reply_markup=keyboard)
    return BP_CONFIRM


async def bp_confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.delete()
    if query.data == "bp_confirm_no":
        await query.message.reply_text("❌ Portfolio build cancelled.")
        context.user_data.clear()
        return ConversationHandler.END

    d = context.user_data
    amount = d["bp_amount"]
    preferences = {
        "risk_appetite": d.get("bp_risk",    "Moderate"),
        "horizon":       d.get("bp_horizon", "Long >5yr"),
        "sector_focus":  d.get("bp_focus",   "All sectors"),
        "sector_avoid":  d.get("bp_avoid",   "None"),
        "stock_count":   d.get("bp_count",   "Auto"),
    }
    context.user_data.clear()

    await query.message.reply_text(
        f"⏳ Screening NSE large & upper mid-cap stocks and building your ₹{amount:,.0f} portfolio…\n"
        "This may take 2–3 minutes.",
    )
    try:
        report_path, input_path, usage_msg = await asyncio.to_thread(
            run_fresh_portfolio_analysis, amount, preferences
        )
        for path, caption in [
            (input_path,  "📋 Input prompt — screened candidates + fundamentals"),
            (report_path, "📄 Fresh portfolio report — open in any Markdown viewer"),
        ]:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    await context.bot.send_document(
                        chat_id=query.message.chat_id,
                        document=f,
                        filename=os.path.basename(path),
                        caption=caption,
                    )
                os.remove(path)
        await query.message.reply_text(usage_msg)
    except Exception as e:
        logger.error("Error in buildPortfolio: %s", e, exc_info=True)
        await query.message.reply_text(f"❌ Error building portfolio: {e}")
    return ConversationHandler.END


async def bp_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ Portfolio build cancelled.")
    return ConversationHandler.END
