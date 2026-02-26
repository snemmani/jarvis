import os
from hmac import new
from re import sub
from tabnanny import check
import trace
from litellm import transcription
import openai
from telegram import Update
import base64
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from bujo.base import OPENAI_MODEL, TELEGRAM_TOKEN, expenses_model, mag_model, llm, check_authorization, WOLFRAM_APP_ID, PC_MAC_ADDRESS, BROADCAST_IP, scheduler, CHAT_ID, openai_model, TEXT_TO_SPEECH_MODEL, portfolio_transactions_model
from bujo.expenses.manage import ExpenseManager
from bujo.mag.manage import MagManager
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
import requests
from langchain_classic.agents import initialize_agent, Tool
from langchain_classic.agents.agent_types import AgentType
from datetime import datetime
import wolframalpha
from wakeonlan import send_magic_packet
import logging
import sys
from typing import List, Dict
import telegram
from apscheduler.triggers.cron import CronTrigger
import json
import ssl
import base64
import yfinance as yf

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = [
    "You are an expert personal assistant that helps me manage my finances and a calendar (which I call MAG).",

    "You have access to the following tools, and depending on my request, you will call the appropriate tool:",

    "1. Expenses Tool – If I talk about expenses, use this tool to add or list my expenses.",
    "2. MAG Tool – If I talk about MAG (my calendar/events), use this tool to add or list my MAG.",
    "3. Wolfram Alpha Tool – If I ask about current events, latest news, mathematical questions, astronomical questions, conversions between units, flight ticket fares, nutrition information of food, stock prices, use this tool.",
    "4. Translation Tool – If I ask for translation:\n   - And I do not specify source/target languages, assume English to Sanskrit.\n   - If I do specify the languages, translate accordingly.",

    "MOST IMPORTANT INSTRUCTIONS:",
    "Always call the appropriate tool based on my latest message. You must never answer directly without invoking a tool first.",
    "When sending a response (either to the LLM for summarization or to me), always return it as a string, not as a JSON object.",
    "Always return tool results in markdown format for readability.",
    "Separate individual items in the tool results with new lines.",
    "Use emojis wherever appropriate to make the response more friendly and visually appealing. 🎯🧮📅💸📰🌐🔢📊🖼️🔤"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def prepend_system_prompt(user_input: str, sys_prompt: str) -> str:
    return f"{sys_prompt}\n\nUser: {user_input}"

def build_chat_messages(user_input: str, sys_prompt: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content':sys_prompt},
        {'role':'human', 'content': user_input}
    ]

expense_manager = ExpenseManager(expenses_model, mag_model)
mag_manager = MagManager(mag_model)
memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)

wolfram_client = wolframalpha.Client(WOLFRAM_APP_ID)

tools = [
    Tool(
        name="Expenses Interaction",
        func=expense_manager.agent_expenses,
        description="Use this tool to fetch expenses based on specific filters.",
        return_direct=True,

    ),
    Tool(
        name="MAG interaction",
        func=mag_manager.agent_mag,
        description="Use this tool to manage MAG.",
        return_direct=True
    ),
    Tool(
        name="Translation Tool",
        func=lambda x: 'This tool must be awaited',
        description="Use this tool to translate from one language to another.",
        coroutine=lambda x: llm.ainvoke([HumanMessage(x)])
    )
]

# Tool function with runtime-bound user_id
async def make_wolfram_alpha_tool(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tool:
    async def query_tool(query: str):
        logger.info(f"Queriying Wolfram Alpha for query: {query}")
        try:
            response = await wolfram_client.aquery(query)
            if hasattr(response, 'pod'):
                for pod in response['pod']:
                    subpod = pod['subpod']
                    if type(subpod) is list:
                        for item in subpod:
                            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=item['img']['@src'], caption=item['@title'])
                    else:
                        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=subpod['img']['@src'], caption=pod['@title'])
            return "Response completed."
        except Exception as e:
            logger.error(f"Error querying Wolfram Alpha: {e}")
            return "Wolfram Alpha cannot handle this request, LLM should handle this if it can."
    
    return Tool(
        name="Wolfram Alpha Tool",
        description="Wolfram Alpha Tool for complex or realtime queries.",
        func=query_tool,
        coroutine=query_tool
    )

# Start command
@check_authorization
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"/start command received from user: {update.effective_user.id}")
    await update.message.reply_text("Hi! I'm your Finances Bot 💳")

@check_authorization
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    await agent_engage(update, context, text)

async def agent_engage(update, context, text):
    # Initialize the tool here
    tools_copy = tools.copy()
    tools_copy.append(await make_wolfram_alpha_tool(update, context))
    agent = initialize_agent(
        tools=tools_copy, 
        llm=llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        memory=memory,
        # handle_parsing_errors=True,
        verbose=True
    )
    logger.info(f"Received chat message from user {update.effective_user.id}: {text}")
    sys_prompt = SYSTEM_PROMPT.copy()
    sys_prompt.append(f'Today\'s date is {datetime.now().strftime("%Y-%m-%d %A")}')
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)
        response = await agent.ainvoke(prepend_system_prompt(text, sys_prompt))
        logger.info(f"Agent response: {response}")
        if 'output' in response and "HERE_IS_IMAGE" in response['output']:
            try:
                images_data = response['output'].split('\n')
                for image in images_data[1:]:
                    title, image_url = image.split('=>')
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_url, caption=title)
                logger.info(f"Sent images to user {update.effective_user.id}: {image_url}")
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                await update.message.reply_text(f"Error generating image: {e}")
        else:
            await update.message.reply_text(
                response['output'],
                parse_mode='markdown',
            )
    except Exception as e:
        logger.error(f"Error in chat handler: {e}")
        await update.message.reply_text(f"An error occurred: {e}")

@check_authorization
async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice.file_id
    new_file = await context.bot.get_file(voice)
    file_path = f"{voice}.ogg"

    await new_file.download_to_drive(file_path)

    with open(file_path, "rb") as audio_file:
        transcript = openai_model.audio.transcriptions.create(
            model=TEXT_TO_SPEECH_MODEL,
            file=audio_file,
        )

    text = transcript.text.strip()
    await agent_engage(update, context, text)

@check_authorization
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    image = update.message.photo[-1]
    text = update.message.caption if update.message.caption else ""
    file_info = await context.bot.get_file(image)
    file_name = os.path.basename(file_info.file_path) if file_info.file_path else f"{image.file_id}.jpg"

    await file_info.download_to_drive(file_name)

    with open(file_name, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = openai_model.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize the image content in a single sentence.\nExcept if it is a transaction.\nIf it is a transaction, Provide response as below message.\nSpent <amount> on <item|whoever the money was sent to> on <date>.\nIf date is not present, assume today\'s date.\nIf amount is not present, assume it is zero.\nIf item is not present, assume it is miscellaneous."},
                {"type": "input_text", "text": "The caption of the Image is: " + text +" so assume the caption is the <item> for this transaction"} if text else {"type": "input_text", "text": "No caption provided."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }]
    )

    await agent_engage(update, context, response.output_text)


@check_authorization
async def wakeUpThePC(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"wakeUpThePC command received from user: {update.effective_user.id}")
    try:
        send_magic_packet(PC_MAC_ADDRESS, ip_address=BROADCAST_IP)
        await update.message.reply_text("🔌 Magic packet sent to wake up the PC.")
        logger.info("Magic packet sent successfully.")
    except requests.RequestException as e:
        logger.error(f"Error sending magic packet: {e}")
        await update.message.reply_text(f"Waking up the PC: {e}")
        

@check_authorization
async def genPass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"GenPass triggered from user: {update.effective_user.id}")
    try:
        translation = str.maketrans('/=+-','abcd')
        num_chars = int(context.args[0]) if context.args else 13
        password = ""
        for iter in range(num_chars//4):
            password += f"{base64.b64encode(ssl.RAND_bytes(13)).decode('utf-8')[:4].translate(translation)}-"
        
        password = password.strip('-')[:num_chars]
        await update.message.reply_text("🔑 Password Generated:")
        await update.message.reply_text(password)
        logger.info("Secure password generated.")
    except requests.RequestException as e:
        logger.error(f"Unable to Generate password: {e}")
        await update.message.reply_text(f"Failed to generate password: {e}")

async def send_mag_message(bot: telegram.Bot):
    """
    Function to send a message to the MAG channel.
    This can be scheduled to run periodically.
    """
    try:
        mag_info_list = mag_manager.mag_model.list(json.dumps({"filters": [f"(Date,eq,exactDate,{datetime.now().strftime('%Y-%m-%d')})"]}))
        if len(mag_info_list) == 0:
            response = "No MAG entries found for today."
            return
        else:
            mag_info = mag_info_list[0]
            response = "Todays's MAG:\n" + \
                f"**📅 Date:** {mag_info['Date']}\n" + \
                f"**🌖 Tithi:** {mag_info['Tithi']}\n" 
            
            if mag_info['Note']:
                response += f"**📝 Note:** {mag_info['Note']}\n"
        # Replace with actual logic to fetch or generate the message
        await bot.send_message(chat_id=CHAT_ID, text=response, parse_mode='markdown')
        logger.info("Scheduled MAG message sent successfully.")
    except Exception as e:
        logger.error(f"Error sending scheduled MAG message: {e}")


@check_authorization
async def get_cmp_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get unique tickers from transactions table
        transactions = portfolio_transactions_model.list()
        tickers = set(tx.get('Ticker') for tx in transactions if tx.get('Ticker'))
        print (transactions)
        if not tickers:
            logger.info("No tickers found in transactions table.")
            return
        
        
        # Fetch current market prices and update transactions
        for ticker in tickers:
            # try:
                data = yf.Ticker(ticker)
                cmp = data.info.get('currentPrice', 0)
                
                # Update all rows with this ticker
                for tx in transactions:
                    if tx.get('Ticker') == ticker:
                        tx['CMP'] = cmp
                        portfolio_transactions_model.update(tx)
                        
                
                logger.info(f"Updated CMP for ticker {ticker}: {cmp}")
            # except Exception as e:
            #     logger.error(f"Error fetching price for ticker {ticker}: {e}")
        
        await context.bot.send_message(chat_id=CHAT_ID, text="✅ CMP values updated for all tickers.", parse_mode='markdown')
    except Exception as e:
        logger.error(f"Error in get_cmp_today: {e}")

@check_authorization
async def get_profit_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_chat_action(telegram.constants.ChatAction.TYPING)

        # ── 1. Fetch all transactions ────────────────────────────────────────────
        transactions = portfolio_transactions_model.list()
        if not transactions:
            await update.message.reply_text("📭 No portfolio transactions found.")
            return

        # ── 2. Fetch USD → INR conversion rate via yfinance ──────────────────────
        usd_inr_ticker = yf.Ticker("USDINR=X")
        usd_to_inr = usd_inr_ticker.info.get("regularMarketPrice") or usd_inr_ticker.fast_info.get("lastPrice", 84.0)
        logger.info(f"USD → INR rate: {usd_to_inr}")

        # ── 3. Aggregate per ticker ──────────────────────────────────────────────
        # ticker_data[ticker] = {
        #   "currency": "INR" | "USD",
        #   "total_bought": float,   # total shares bought
        #   "total_sold": float,     # total shares sold
        #   "buy_cost_inr": float,   # weighted cost basis in INR
        #   "sell_proceeds_inr": float,
        #   "cmp_inr": float,        # latest CMP in INR
        # }
        ticker_data: Dict[str, Dict] = {}

        for tx in transactions:
            ticker    = tx.get("Ticker", "").strip()
            tx_type   = tx.get("TransactionType", "").strip()   # "Buy" or "Sell"
            shares    = float(tx.get("NoOfShares") or 0)
            cost      = float(tx.get("CostPerShare") or 0)
            cmp       = float(tx.get("CMP") or 0)

            if not ticker or shares == 0:
                continue

            is_inr    = ticker.upper().endswith(".NS")
            currency  = "INR" if is_inr else "USD"
            fx        = 1.0 if is_inr else usd_to_inr

            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    "currency": currency,
                    "total_bought": 0.0,
                    "total_sold": 0.0,
                    "buy_cost_inr": 0.0,
                    "sell_proceeds_inr": 0.0,
                    "cmp_inr": cmp * fx,
                }

            # Always keep the latest CMP (last row wins – same as get_cmp_today logic)
            if cmp:
                ticker_data[ticker]["cmp_inr"] = cmp * fx

            if tx_type == "Buy":
                ticker_data[ticker]["total_bought"]  += shares
                ticker_data[ticker]["buy_cost_inr"]  += shares * cost * fx
            elif tx_type == "Sell":
                ticker_data[ticker]["total_sold"]     += shares
                ticker_data[ticker]["sell_proceeds_inr"] += shares * cost * fx

        # ── 4. Classify tickers & compute P&L ───────────────────────────────────
        open_tickers:   List[Dict] = []
        closed_tickers: List[Dict] = []

        for ticker, d in ticker_data.items():
            net_shares   = d["total_bought"] - d["total_sold"]
            avg_cost_inr = (d["buy_cost_inr"] / d["total_bought"]) if d["total_bought"] else 0.0
            cmp_inr      = d["cmp_inr"]

            if net_shares > 0:
                # ── OPEN position ────────────────────────────────────────────────
                current_value_inr  = net_shares * cmp_inr
                invested_value_inr = net_shares * avg_cost_inr
                unrealised_pl      = current_value_inr - invested_value_inr
                unrealised_pct     = (unrealised_pl / invested_value_inr * 100) if invested_value_inr else 0.0

                # Realised P&L from partial sells (if any)
                realised_pl = d["sell_proceeds_inr"] - (d["total_sold"] * avg_cost_inr)

                open_tickers.append({
                    "ticker":           ticker,
                    "currency":         d["currency"],
                    "net_shares":       net_shares,
                    "avg_cost_inr":     avg_cost_inr,
                    "cmp_inr":          cmp_inr,
                    "invested_inr":     invested_value_inr,
                    "current_inr":      current_value_inr,
                    "unrealised_pl":    unrealised_pl,
                    "unrealised_pct":   unrealised_pct,
                    "realised_pl":      realised_pl,
                })
            else:
                # ── CLOSED position (fully sold) ─────────────────────────────────
                realised_pl  = d["sell_proceeds_inr"] - d["buy_cost_inr"]
                realised_pct = (realised_pl / d["buy_cost_inr"] * 100) if d["buy_cost_inr"] else 0.0

                closed_tickers.append({
                    "ticker":        ticker,
                    "currency":      d["currency"],
                    "buy_cost_inr":  d["buy_cost_inr"],
                    "sell_inr":      d["sell_proceeds_inr"],
                    "realised_pl":   realised_pl,
                    "realised_pct":  realised_pct,
                })

        # ── 5. Build the Telegram message ────────────────────────────────────────
        def fmt_inr(val: float) -> str:
            """Format a rupee value with ₹ sign and comma separators."""
            return f"₹{val:,.2f}"

        def fmt_usd(val: float) -> str:
            """Format a dollar value with $ sign and comma separators."""
            return f"${val:,.2f}"

        def pl_emoji(val: float) -> str:
            return "🟢" if val >= 0 else "🔴"

        def build_section(
            section_open: List[Dict],
            section_closed: List[Dict],
            is_usd: bool,
        ) -> List[str]:
            """Build lines for one currency bucket (INR or USD)."""
            fmt      = fmt_usd if is_usd else fmt_inr
            # For USD tickers the stored values are already in INR (multiplied by fx),
            # but we want to display them in USD for clarity.
            fx_div   = usd_to_inr if is_usd else 1.0
            sec_lines: List[str] = []

            # ── Open ─────────────────────────────────────────────────────────────
            if section_open:
                inv_total  = sum(t["invested_inr"] for t in section_open)
                cur_total  = sum(t["current_inr"]  for t in section_open)
                unreal_tot = cur_total - inv_total
                real_open  = sum(t["realised_pl"]  for t in section_open)

                sec_lines.append(f"📂 *Open* ({len(section_open)})")
                sec_lines.append("─────────────────────────────")

                for t in sorted(section_open, key=lambda x: x["unrealised_pl"], reverse=True):
                    emoji = pl_emoji(t["unrealised_pl"])
                    avg   = t["avg_cost_inr"]  / fx_div
                    cmp   = t["cmp_inr"]       / fx_div
                    inv   = t["invested_inr"]  / fx_div
                    cur   = t["current_inr"]   / fx_div
                    upl   = t["unrealised_pl"] / fx_div
                    rpl   = t["realised_pl"]   / fx_div
                    sec_lines.append(
                        f"{emoji} *{t['ticker']}*\n"
                        f"   Shares: `{t['net_shares']:.4f}` | Avg Cost: {fmt(avg)}\n"
                        f"   CMP: {fmt(cmp)} | Invested: {fmt(inv)}\n"
                        f"   Current: {fmt(cur)} | "
                        f"Unrealised: {fmt(upl)} ({t['unrealised_pct']:+.2f}%)"
                        + (f"\n   Realised (partial): {fmt(rpl)}" if t["realised_pl"] != 0 else "")
                    )

                sec_lines.append("─────────────────────────────")
                sec_lines.append(
                    f"📌 *Open Totals*\n"
                    f"   Invested: {fmt(inv_total / fx_div)}\n"
                    f"   Current:  {fmt(cur_total / fx_div)}\n"
                    f"   {pl_emoji(unreal_tot)} Unrealised P&L: {fmt(unreal_tot / fx_div)} "
                    f"({(unreal_tot / inv_total * 100) if inv_total else 0:+.2f}%)"
                    + (f"\n   Realised (partial sells): {fmt(real_open / fx_div)}" if real_open else "")
                )
            else:
                sec_lines.append("📂 *Open Positions:* None")

            # ── Closed ───────────────────────────────────────────────────────────
            if section_closed:
                buy_tot  = sum(t["buy_cost_inr"] for t in section_closed)
                sell_tot = sum(t["sell_inr"]      for t in section_closed)
                real_tot = sell_tot - buy_tot

                sec_lines.append("")
                sec_lines.append(f"✅ *Closed* ({len(section_closed)})")
                sec_lines.append("─────────────────────────────")

                for t in sorted(section_closed, key=lambda x: x["realised_pl"], reverse=True):
                    emoji = pl_emoji(t["realised_pl"])
                    sec_lines.append(
                        f"{emoji} *{t['ticker']}*\n"
                        f"   Invested: {fmt(t['buy_cost_inr'] / fx_div)} | Sold: {fmt(t['sell_inr'] / fx_div)}\n"
                        f"   Realised P&L: {fmt(t['realised_pl'] / fx_div)} ({t['realised_pct']:+.2f}%)"
                    )

                sec_lines.append("─────────────────────────────")
                sec_lines.append(
                    f"📌 *Closed Totals*\n"
                    f"   Total Invested: {fmt(buy_tot / fx_div)}\n"
                    f"   Total Sold:     {fmt(sell_tot / fx_div)}\n"
                    f"   {pl_emoji(real_tot)} Realised P&L: {fmt(real_tot / fx_div)} "
                    f"({(real_tot / buy_tot * 100) if buy_tot else 0:+.2f}%)"
                )
            else:
                sec_lines.append("\n✅ *Closed Positions:* None")

            return sec_lines

        # ── Split by currency ─────────────────────────────────────────────────────
        inr_open   = [t for t in open_tickers   if t["currency"] == "INR"]
        inr_closed = [t for t in closed_tickers if t["currency"] == "INR"]
        usd_open   = [t for t in open_tickers   if t["currency"] == "USD"]
        usd_closed = [t for t in closed_tickers if t["currency"] == "USD"]

        lines: List[str] = []

        # ── Header ───────────────────────────────────────────────────────────────
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("📊 *Portfolio P&L Summary*")
        lines.append(f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
        lines.append(f"💱 USD → INR: ₹{usd_to_inr:.2f}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ── 🇮🇳 INR Section ───────────────────────────────────────────────────────
        lines.append("")
        lines.append("🇮🇳 *Indian Portfolio (INR)*")
        lines.extend(build_section(inr_open, inr_closed, is_usd=False))

        # INR section subtotal
        inr_invested = sum(t["invested_inr"] for t in inr_open)   + sum(t["buy_cost_inr"] for t in inr_closed)
        inr_current  = sum(t["current_inr"]  for t in inr_open)   + sum(t["sell_inr"]     for t in inr_closed)
        inr_pl       = inr_current - inr_invested
        lines.append("")
        lines.append(
            f"🏦 *INR Net*: {fmt_inr(inr_invested)} → {fmt_inr(inr_current)}  "
            f"{pl_emoji(inr_pl)} {fmt_inr(inr_pl)} "
            f"({(inr_pl / inr_invested * 100) if inr_invested else 0:+.2f}%)"
        )

        # ── 🇺🇸 USD Section ───────────────────────────────────────────────────────
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("🇺🇸 *US Portfolio (USD)*")
        lines.extend(build_section(usd_open, usd_closed, is_usd=True))

        # USD section subtotal (shown in both USD and INR equivalent)
        usd_invested_inr = sum(t["invested_inr"] for t in usd_open)   + sum(t["buy_cost_inr"] for t in usd_closed)
        usd_current_inr  = sum(t["current_inr"]  for t in usd_open)   + sum(t["sell_inr"]     for t in usd_closed)
        usd_pl_inr       = usd_current_inr - usd_invested_inr
        lines.append("")
        lines.append(
            f"🏦 *USD Net*: {fmt_usd(usd_invested_inr / usd_to_inr)} → {fmt_usd(usd_current_inr / usd_to_inr)}  "
            f"{pl_emoji(usd_pl_inr)} {fmt_usd(usd_pl_inr / usd_to_inr)} "
            f"({(usd_pl_inr / usd_invested_inr * 100) if usd_invested_inr else 0:+.2f}%)\n"
            f"   _(≈ {fmt_inr(usd_invested_inr)} → {fmt_inr(usd_current_inr)}, P&L {fmt_inr(usd_pl_inr)})_"
        )

        # ── Grand Total (everything in INR) ──────────────────────────────────────
        all_invested = inr_invested + usd_invested_inr
        all_current  = inr_current  + usd_current_inr
        grand_pl     = all_current  - all_invested

        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(
            f"💼 *Grand Total (INR equivalent)*\n"
            f"   Total Invested: {fmt_inr(all_invested)}\n"
            f"   Total Value:    {fmt_inr(all_current)}\n"
            f"   {pl_emoji(grand_pl)} Overall P&L: {fmt_inr(grand_pl)} "
            f"({(grand_pl / all_invested * 100) if all_invested else 0:+.2f}%)"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        message = "\n".join(lines)

        await update.message.reply_text(message, parse_mode="Markdown")
        logger.info("P&L summary sent successfully.")

    except Exception as e:
        logger.error(f"Error in get_profit_loss: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error generating P&L report: {e}")


async def setup_scheduler(application):
    scheduler.add_job(
        send_mag_message,
        CronTrigger(hour="8", minute="0", day="*", month="*", day_of_week="*"),
        args=[application.bot]
    )
    scheduler.start()
    logger.info("🕒 Scheduler started for sending calendar at 8:00.")

# Main runner
if __name__ == '__main__':
    logger.info("Starting the Telegram bot application.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(setup_scheduler).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, voice))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, image))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("wakeTheBeast", wakeUpThePC))
    app.add_handler(CommandHandler("genPass", genPass, has_args=1))
    app.add_handler(CommandHandler("updateTicker", get_cmp_today))
    app.add_handler(CommandHandler("getProfitLoss", get_profit_loss))
    print("🤖 Bot is running...")
    logger.info("🤖 Bot is running...")
    app.run_polling()
