<p align="center">
  <h1 align="center">🤖 JARVIS Bot</h1>
  <p align="center">
    <strong>Your AI-powered personal assistant on Telegram</strong>
  </p>
  <p align="center">
    Manage expenses · Track your calendar · Monitor your portfolio · Dispatch Claude AI · And more
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#nocodb-table-structure">NocoDB Tables</a> •
    <a href="#environment-variables">Environment Variables</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#commands">Commands</a> •
    <a href="#implementation-details">Implementation Details</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## ✨ Overview

JARVIS Bot is a personal Telegram bot that acts as an intelligent assistant for managing daily finances, a personal calendar (called **MAG** — Month/Day at a Glance), and an investment portfolio. It uses **OpenAI GPT** via **LangChain/LangGraph** to understand natural language, route requests to the right tools, and respond in a friendly, emoji-rich markdown format.

Send it a text message, a voice note, or even a photo of a receipt — JARVIS figures out what you need and gets it done. It also embeds **Claude Code CLI** as an agentic dispatcher for deep research and general AI tasks.

---

## 🚀 Features

### 💸 Expense Management
- **Add expenses** in natural language — *"Spent 500 on groceries today"*
- **Send a photo** of a receipt or payment screenshot and JARVIS extracts the amount, item, and date using GPT vision
- **List & filter expenses** by date range, week, month, or specific day
- Expenses are automatically **linked to the corresponding MAG** (calendar) entry
- All amounts displayed in **₹ (Indian Rupees)**

### 📅 MAG — Personal Calendar
- **View your calendar** — *"Show me MAG for this week"*
- **Update notes & exercise tracking** — *"I completed my exercise today"*
- Tracks: Date, Tithi (Telugu calendar), Notes, and Exercise status
- **Scheduled daily briefing** — automatically sends today's MAG at **8:00 AM IST**

### 📊 Portfolio Tracker
- Track **Indian (NSE)** and **US stock** positions across named portfolios
- **Live CMP updates** via [yfinance](https://github.com/ranaroussi/yfinance) — scheduled every weekday at **8:15 AM IST**
- **Profit & Loss reports** with unrealised/realised P&L, grouped by portfolio and currency
- Automatic **USD → INR** conversion for US-listed stocks

### 🧠 Claude AI Dispatcher
- **`/claudeApi <prompt>`** — dispatches any prompt to Claude Code CLI, maintains a **per-user conversation session** across calls (context is preserved between messages)
- **`/portfolioSuggest`** — fetches all portfolio transactions, combines them with a detailed prompt, runs a **deep web-research** session via Claude (WebSearch + WebFetch enabled), and delivers a **PDF research report** directly to your Telegram chat

### 🔍 Knowledge & Translation
- **Wolfram Alpha integration** — math, science, conversions, nutrition, stock prices, flight fares, and more (returns visual pods as images)
- **Translation** — translate text between languages (defaults to English → Sanskrit)
- **Conversational memory** — maintains context across messages per user session

### 🎤 Voice & Vision Input
- **Voice messages** — transcribed via OpenAI Whisper, then processed as text
- **Image analysis** — GPT-4 Vision extracts transaction details from receipt/payment photos

### 🔧 Utility Commands
- `/wakeTheBeast` — Send a Wake-on-LAN magic packet to power on your PC remotely
- `/genPass <length>` — Generate a cryptographically secure random password
- `/ddns update` — Update your No-IP DDNS hostname with the current public IPv6
- `/ddns block` — Point the hostname to a mangled IPv6 to cut external access at will

### 🔒 Security
- **User authorization** — only whitelisted Telegram user IDs can interact with the bot
- Unauthorized messages are forwarded to the owner for awareness
- Container runs as a **non-root `bot` user** for reduced attack surface

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Telegram User                           │
│              Text / Voice / Photo / Commands                   │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                  bujo-bot.py  (Main Bot)                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Top-Level LangGraph ReAct Agent (GPT)           │  │
│  │                                                          │  │
│  │  Expenses_  MAG_    Portfolio_ Translation_ Wolfram_     │  │
│  │  Interact.  Interact. Tool     Tool         Alpha_Tool   │  │
│  │           + Expense_Analytics_Tool                       │  │
│  └────┬──────────┬──────────┬───────────────────────────────┘  │
│       ▼          ▼          ▼                                   │
│  ┌──────────────────────────────────────┐                      │
│  │            Manager Layer             │                      │
│  │  ExpenseManager  │  MagManager  │    │                      │
│  │  PortfolioManager                    │                      │
│  │  (each: own LangGraph ReAct agent)   │                      │
│  └──────────────────┬───────────────────┘                      │
│                     ▼                                          │
│  ┌──────────────────────────────────────┐                      │
│  │       Model Layer (NocoDB REST)      │                      │
│  │  Expenses │ MAG │ PortfolioTransact. │                      │
│  │  (BaseNocoDB → paginated REST API)   │                      │
│  └──────────────────────────────────────┘                      │
│                                                                │
│  ┌──────────────────────────────────────┐                      │
│  │       Claude CLI Dispatcher          │                      │
│  │  /claudeApi  →  _run_claude()        │                      │
│  │  /portfolioSuggest  →  _run_claude() │                      │
│  │  (per-user session via --resume)     │                      │
│  └──────────────────────────────────────┘                      │
│                                                                │
│  APScheduler (8:00 AM MAG briefing, 8:15 AM CMP update)       │
└────────────────────────────────────────────────────────────────┘
                │                          │
                ▼                          ▼
       ┌─────────────────┐       ┌──────────────────────┐
       │     NocoDB      │       │  Claude Code CLI      │
       │  (Database API) │       │  (Node.js, OAuth via  │
       └─────────────────┘       │   ~/.claude mount)    │
                                 └──────────────────────┘
```

### Key Design Decisions

- **Multi-agent architecture** — A top-level ReAct agent routes to six specialized tools. Expenses, MAG, and Portfolio each run their own inner LangGraph ReAct agent with a domain-specific system prompt and tool set.
- **NocoDB as backend** — All persistent data lives in NocoDB tables accessed via its REST API. This lets you view and edit records directly in the NocoDB UI without touching the bot.
- **LangGraph with memory** — `MemorySaver` checkpointers maintain per-user conversation state. The top-level agent uses `thread_id = "user_<id>"` so each user gets isolated history.
- **Claude CLI sessions** — Claude Code CLI is invoked as an async subprocess with `--output-format json`. The returned `session_id` is stored in `_claude_sessions[user_id]` and passed as `--resume` on subsequent calls, giving each Telegram user their own continuous Claude conversation.
- **APScheduler** — Cron-based async scheduler for daily MAG briefings and weekday CMP stock price updates, both sent to the configured `CHAT_ID`.

---

## 🗄️ NocoDB Table Structure

Three tables are required. Create them in your NocoDB project and copy the table IDs into your `.env`.

### Table 1: Expenses

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Date` | Date | ✅ | Format: `YYYY-MM-DD` |
| `Item` | Single line text | ✅ | Description of the expense |
| `Amount` | Number (decimal) | ✅ | Amount in ₹ |
| `MAG` | Link to MAG table | optional | Relationship field — note the **Link Field ID** for `NOCODB_EXPENSES_MAG_LINK_ID` |

> After creating the relationship between Expenses and MAG, find the link field ID in the NocoDB API docs (it appears as the field ID in the `/api/v2/tables/{table_id}/links/` endpoint).

---

### Table 2: MAG (Month/Day at a Glance)

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Date` | Date | ✅ | Format: `YYYY-MM-DD` — one row per calendar day |
| `Tithi` | Single line text | optional | Telugu/Hindu calendar day name |
| `Note` | Long text | optional | Free-form daily note — LLM-updatable |
| `Exercise` | Checkbox or text | optional | Exercise completion for the day — LLM-updatable |
| `Expenses` | Link to Expenses | optional | Reverse side of the Expenses → MAG relationship |

> Pre-populate this table with one row per day (or generate rows as needed). The bot's `find_by_date` lookup queries by exact `Date` match.

---

### Table 3: PortfolioTransactions

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Ticker` | Single line text | ✅ | Stock symbol e.g. `PFC.NS`, `AAPL` — append `.NS` for NSE stocks |
| `TransactionType` | Single line text | ✅ | `Buy` or `Sell` (capitalised) |
| `NoOfShares` | Number (decimal) | ✅ | Number of shares |
| `CostPerShare` | Number (decimal) | ✅ | Price per share in native currency (₹ for `.NS`, $ for US) |
| `Date` | Date | ✅ | Transaction date, format `YYYY-MM-DD` |
| `Portfolio` | Single line text | optional | Portfolio name e.g. `LT`, `Default` — defaults to `Default` |
| `CMP` | Number (decimal) | optional | Current Market Price — updated automatically by the CMP scheduler |

---

## 🔑 Environment Variables

Create a `.env` file in the project root (copy `.env.example` if provided):

```env
# Telegram
TELEGRAM_TOKEN=your_bot_token_from_botfather
TELEGRAM_USER_ID=your_telegram_numeric_user_id
CHAT_ID=your_telegram_chat_id          # used for scheduled messages and forwarding unauthorized requests

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o                    # or gpt-4-turbo etc.
TEXT_TO_SPEECH_MODEL=whisper-1

# NocoDB
NOCODB_BASE_URL=http://your-nocodb-host:8080
NOCODB_API_TOKEN=your_nocodb_api_token
NOCODB_EXPENSES_TABLE_ID=md_xxxxxxxx
NOCODB_MAG_TABLE_ID=md_yyyyyyyy
NOCODB_TRANSACTIONS_TABLE_ID=md_zzzzzzzz
NOCODB_EXPENSES_MAG_LINK_ID=link_field_id   # ID of the link field connecting Expenses → MAG

# Wolfram Alpha
WOLFRAM_APP_ID=your_wolfram_app_id

# Wake-on-LAN
PC_MAC_ADDRESS=AA:BB:CC:DD:EE:FF
BROADCAST_IP=192.168.1.255

# No-IP DDNS
NOIP_USERNAME=your_noip_username
NOIP_PASSWORD=your_noip_password
NOIP_HOSTNAME=yourhost.ddns.net

# (optional — present in config but currently unused)
SERP_API_KEY=your_serp_api_key
```

---

## 📁 Project Structure

```
jarvis/
├── bujo/
│   ├── bujo-bot.py              # Main entry point — Telegram handlers, top-level agent,
│   │                            # Claude CLI dispatcher, APScheduler setup
│   ├── base.py                  # Config, env vars, LLM init, auth decorator,
│   │                            # model/scheduler instances
│   ├── expenses/
│   │   └── manage.py            # ExpenseManager — inner ReAct agent for expense CRUD
│   ├── mag/
│   │   └── manage.py            # MagManager — inner ReAct agent for calendar management
│   ├── portoflio/
│   │   └── manage.py            # PortfolioManager — portfolio CRUD, CMP updates, P&L reports
│   ├── models/
│   │   ├── base.py              # BaseNocoDB — generic paginated REST client
│   │   ├── expenses.py          # Expenses model — CRUD + MAG link management
│   │   ├── mag.py               # MAG model — CRUD + date-based lookup
│   │   └── portfolio_transactions.py  # PortfolioTransactions model — CRUD + CMP patch
│   └── analytics/
│       └── charts.py            # spending_pie_chart / spending_bar_chart (matplotlib)
├── tests/
├── Dockerfile                   # Python 3.13-slim + Node.js 22 + Claude Code CLI,
│                                # runs as non-root `bot` user
├── requirements.txt
├── .env                         # Not committed — see Environment Variables above
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| **Bot Framework** | [python-telegram-bot 22.7](https://python-telegram-bot.org/) |
| **LLM** | OpenAI GPT via [LangChain](https://www.langchain.com/) |
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) (ReAct agents with MemorySaver) |
| **Speech-to-Text** | OpenAI Whisper |
| **Vision** | OpenAI GPT-4 Vision |
| **Database** | [NocoDB](https://nocodb.com/) REST API v2 |
| **Knowledge Engine** | [Wolfram Alpha](https://www.wolframalpha.com/) |
| **Stock Data** | [yfinance](https://github.com/ranaroussi/yfinance) |
| **AI Dispatcher** | [Claude Code CLI](https://docs.anthropic.com/claude-code) (`@anthropic-ai/claude-code`) |
| **PDF Generation** | [fpdf2](https://py-fpdf2.readthedocs.io/) |
| **Scheduling** | [APScheduler](https://apscheduler.readthedocs.io/) (async cron) |
| **Wake-on-LAN** | [wakeonlan](https://pypi.org/project/wakeonlan/) |
| **Containerization** | Docker (Python 3.13-slim + Node.js 22) |

---

## 🚀 Getting Started

### Prerequisites

- Docker installed on the host
- A running [NocoDB](https://nocodb.com/) instance with the three tables created (see [NocoDB Table Structure](#nocodb-table-structure))
- A Telegram bot token from [@BotFather](https://t.me/botfather)
- OpenAI API key
- Wolfram Alpha App ID (free tier available)
- Claude Code CLI authenticated on the **host machine** (run `claude` once to log in via browser)

### 1. Clone and configure

```bash
git clone https://github.com/snemmani/jarvis.git
cd jarvis
cp .env.example .env   # then fill in all values
```

### 2. Authenticate Claude Code on the host

```bash
claude   # follow the browser login prompt once — credentials saved to ~/.claude/
```

### 3. Build the Docker image

```bash
docker build -t jarvis .
```

### 4. Run the container

```bash
docker run -d \
  --network=host \
  --name jarvis \
  -v ~/.claude:/home/bot/.claude \
  -v ~/dev/ai-prompts:/home/bot/ai-prompts \
  --env-file .env \
  jarvis
```

| Volume mount | Purpose |
|---|---|
| `~/.claude:/home/bot/.claude` | Claude Code CLI OAuth credentials |
| `~/dev/ai-prompts:/home/bot/ai-prompts` | Investment analysis prompt files (e.g. `GrahamPrompt.md`) |

> The container runs as the non-root user `bot` (home `/home/bot`). The `.claude` directory must be owned/readable by that user — mounting your host `~/.claude` directly works if the UID matches or Docker runs with sufficient privileges.

### Rebuild after code changes

```bash
docker build -t jarvis . && docker run -d --network=host --name jarvis \
  -v ~/.claude:/home/bot/.claude \
  -v ~/dev/ai-prompts:/home/bot/ai-prompts \
  --env-file .env jarvis
```

---

## 💬 Commands

| Command | Description |
|---|---|
| `/start` | Greeting message |
| `/claudeApi <prompt>` | Dispatch a prompt to Claude Code CLI; session is preserved per user across calls |
| `/portfolioSuggest` | Deep portfolio research: fetches all transactions + GrahamPrompt.md, runs web-researched Claude analysis, delivers a PDF report |
| `/getProfitLoss` | Generate a full unrealised/realised P&L report across all portfolios |
| `/updateTicker` | Manually trigger a CMP update for all tickers via yfinance |
| `/ddns update` | Update No-IP hostname to your current public IPv6 |
| `/ddns block` | Mangle the last hex digit of your IPv6 to cut external access |
| `/wakeTheBeast` | Send a Wake-on-LAN magic packet to power on your PC |
| `/genPass <length>` | Generate a cryptographically secure random password |

**Natural language (no command needed):**

| What you say | What happens |
|---|---|
| "Spent 500 on groceries today" | Expense added and linked to today's MAG |
| "Show me expenses for last week" | Lists filtered expenses |
| "Show me MAG for this week" | Returns this week's calendar entries |
| "I completed my exercise today" | Updates Exercise field on today's MAG row |
| "Bought 50 INFY.NS at 1500 today in LT portfolio" | Adds a Buy transaction to PortfolioTransactions |
| "Show my portfolio for March" | Lists transactions with date filter |
| "What is the mass of the sun?" | Queries Wolfram Alpha, returns image pods |
| "Translate hello to Sanskrit" | Calls GPT for translation |
| "Show me spending by category for April" | Generates a pie chart via matplotlib |

---

## ⚙️ Implementation Details

### Agent Routing

Every text message, voice transcription, or image analysis result flows into `agent_engage()`. This builds a fresh LangGraph ReAct agent from `_static_tools` + two dynamically-created tools (`Wolfram_Alpha_Tool`, `Expense_Analytics_Tool` — dynamic because they need the Telegram `update`/`context` objects to send photos). The agent uses a `MemorySaver` with `thread_id = "user_<telegram_id>"` for per-user conversation history.

The top-level system prompt instructs the agent to always call a tool; it never responds directly.

### Manager Sub-Agents

Each domain manager (Expenses, MAG, Portfolio) owns its own inner LangGraph ReAct agent with a domain-specific system prompt. When the top-level agent calls e.g. `Expenses_Interaction("Spent 500 on groceries")`, it invokes `ExpenseManager.agent_expenses()` which runs the inner agent — this two-level design keeps system prompts small and focused.

### NocoDB REST Client

`BaseNocoDB` wraps NocoDB's v2 REST API. All list operations use `_paginated_list()` which iterates pages (1000 rows per page) until `PageInfo.isLastPage` is true — ensuring the full dataset is always returned regardless of size.

Filters are passed as NocoDB query strings, e.g.:
```
(Date,ge,exactDate,2025-03-01)
(Date,lt,exactDate,2025-04-01)
(Ticker,eq,text,PFC.NS)
```

### Claude CLI Dispatcher

`_run_claude(prompt, user_id, allowed_tools, timeout)` is the shared async helper for both `/claudeApi` and `/portfolioSuggest`:

1. Builds the subprocess command: `claude -p <prompt> --output-format json [--allowedTools ...] [--resume <session_id>]`
2. Runs it with `asyncio.create_subprocess_exec` (no shell, safe argument passing)
3. Applies `asyncio.wait_for` with a configurable timeout
4. Parses the JSON output to extract `result` and `session_id`
5. Stores `session_id` in `_claude_sessions[user_id]` — passed as `--resume` on the next call so the Claude conversation continues where it left off
6. On error (`is_error: true`), clears the stored session so the next call starts fresh

`_keep_typing()` runs as a background `asyncio.Task` during any Claude invocation, refreshing the Telegram "typing..." indicator every 4 seconds.

### `/portfolioSuggest` Flow

```
1. Read ~/dev/ai-prompts/GrahamPrompt.md from the mounted volume
2. Fetch all rows from PortfolioTransactions via NocoDB (no filter — full history)
3. Serialize transactions as CSV rows
4. Build a combined prompt: GrahamPrompt framework + CSV data + web-research instruction
5. Call _run_claude() with allowed_tools="WebSearch,WebFetch", timeout=300s
6. Convert the markdown report to PDF via _report_to_pdf() (fpdf2, latin-1 safe)
7. Send as a Telegram document; fall back to long-text if PDF fails
```

### P&L Calculation

`PortfolioManager.get_profit_loss_report()` groups transactions by `(Ticker, Portfolio)`. For each group it computes weighted average cost, net shares (buys minus sells), current value using the stored CMP, and unrealised/realised P&L. US stocks (no `.NS` suffix) are converted to INR using the live `USDINR=X` rate from yfinance.

### Scheduled Jobs

| Time (IST) | Days | Job |
|---|---|---|
| 08:00 | Daily | Send today's MAG entry to `CHAT_ID` |
| 08:15 | Mon–Fri | Fetch CMP for all tickers via yfinance, patch every PortfolioTransactions row |

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ as a personal productivity companion
</p>
