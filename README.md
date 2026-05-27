<p align="center">
  <h1 align="center">JARVIS Bot</h1>
  <p align="center">
    <strong>Your AI-powered personal assistant on Telegram</strong>
  </p>
  <p align="center">
    Manage expenses · Track MAG · Monitor portfolios · Generate rebalance plans
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

## Overview

JARVIS Bot is a personal Telegram bot for managing daily expenses, a personal calendar called **MAG** (Month/Day at a Glance), and investment portfolios. It uses **OpenAI** through **LangChain/LangGraph** to route natural-language messages into focused tools, then replies in Telegram-friendly Markdown.

Send it a text message, a voice note, a receipt/payment screenshot, or a broker trade screenshot. JARVIS parses the input, calls the right manager, stores data in NocoDB, and sends back a concise result.

---

## Features

### Expense Management

- Add expenses in natural language, for example: `Spent 500 on groceries today`
- Parse receipt, payment, and UPI screenshots with OpenAI vision
- List expenses by day, week, month, or arbitrary date ranges
- Link expenses to the matching MAG row through the configured NocoDB relation
- Generate spending charts: category pie/bar, daily bar/line, weekly bar, and monthly bar
- Support include/exclude terms for chart requests, such as `May expenses except Home Loan`

### MAG Calendar

- View MAG entries by day, week, month, or date range
- Update daily notes and exercise completion from natural language
- Track Date, Day, Tithi, Note, Exercise, and linked Expenses
- Send a scheduled daily MAG briefing at 08:00 IST

### Portfolio Tracking

- Track Indian and US stock positions across named portfolios
- Record buys, sells, deposits, and withdrawals from text or trade screenshots
- Store optional transaction notes for thesis, triggers, caution points, or rationale
- Automatically create offsetting `CASH` ledger rows for stock buys and sells
- Convert US-listed positions to INR for dashboards and P&L reporting
- Update CMP values through yfinance on schedule or via `/updateTicker`
- Show an inline Telegram dashboard with overview, holdings, cash, risk, and rebalance views

### Portfolio Research & Alerts

- Generate prompt-gated rebalance reports through `/rebalanceRecommendations`
- Review the full model input document before forwarding it to OpenAI
- Build fresh portfolio plans through `/buildPortfolio` using amount, risk, horizon, sectors, and stock count
- Run portfolio alerts for concentration, negative cash, missing CMP, thesis-note triggers, and model-assisted risk notes
- Create, modify, and cancel ticker price alerts through `/setAlert` and `/listAlerts`

### Knowledge, Voice, And Vision

- Query Wolfram Alpha for math, science, conversions, nutrition, astronomy, and other factual pods
- Translate text with GPT, defaulting to English to Sanskrit when no language pair is supplied
- Transcribe voice messages with OpenAI Whisper before routing them through the same agent
- Classify images as expenses, portfolio transactions, or general images before routing the parsed result

### Utilities And Security

- `/wakeTheBeast` calls a local relay to send a Wake-on-LAN packet
- `/ddns update` and `/ddns block` call the relay for No-IP DDNS control
- `/genPass <length>` creates a cryptographically secure password
- Only whitelisted Telegram users can use the bot; unauthorized messages are forwarded to `CHAT_ID`
- Docker runs the app as a non-root `bot` user

---

## Architecture

```text
Telegram user
  │
  ├─ Text ───────────────┐
  ├─ Voice → Whisper ────┤
  └─ Photo → Vision ─────┤
                         ▼
                 bujo/app.py
       python-telegram-bot handlers/callbacks
                         │
                         ▼
                 bujo/agent.py
        top-level LangGraph ReAct agent
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
 ExpenseManager      MagManager     PortfolioManager
 inner ReAct agent   inner agent     inner ReAct agent
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                 NocoDB REST models
 Expenses · MAG · PortfolioTransactions · PriceAlerts
                         │
                         ▼
                   NocoDB API v2

APScheduler runs daily MAG, CMP refresh, portfolio alerts,
price alerts, and monthly rebalance jobs.
```

`bujo/bujo-bot.py` is a tiny executable wrapper that imports `bujo.app.run()`. The actual Telegram application wiring lives in `bujo/app.py`.

### Key Design Decisions

- **Top-level routing agent**: `agent_engage()` builds a LangGraph ReAct agent with static domain tools plus dynamic Wolfram and analytics tools that can send Telegram photos.
- **Manager sub-agents**: Expenses, MAG, and Portfolio each own a focused inner ReAct agent with a domain-specific prompt and tool set.
- **NocoDB backend**: Persistent data is stored in NocoDB tables accessed through REST API v2.
- **Per-user memory**: The top-level agent uses `MemorySaver` with `thread_id = "user_<telegram_id>"`.
- **Scheduler**: APScheduler attaches jobs during Telegram `post_init` and sends scheduled outputs to `CHAT_ID`.
- **Model compatibility patches**: `bujo/base.py` patches LangChain/OpenAI edge cases around tool-call arguments and unsupported stop sequences.

---

## NocoDB Table Structure

Four tables are required. Create them in your NocoDB project and copy the table IDs into `.env`.

### Table 1: Expenses

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Date` | Date | yes | Format: `YYYY-MM-DD` |
| `Item` | Single line text | yes | Expense description or recipient |
| `Amount` | Number | yes | Amount in INR |
| `MAG` | Link to MAG table | optional | Relationship field used by `NOCODB_EXPENSES_MAG_LINK_ID` |

After creating the Expenses-to-MAG relation, find the relation/link field ID in the NocoDB API metadata and use it for `NOCODB_EXPENSES_MAG_LINK_ID`.

### Table 2: MAG

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Date` | Date | yes | One row per calendar day |
| `Day` | Text | optional | Weekday or label |
| `Tithi` | Text | optional | Telugu/Hindu calendar day name |
| `Note` | Long text | optional | Free-form daily note |
| `Exercise` | Checkbox/boolean | optional | Exercise completion |
| `Expenses` | Link to Expenses | optional | Reverse relation from Expenses |

Pre-populate MAG with one row per day. The bot links expenses by exact `Date` lookup.

### Table 3: PortfolioTransactions

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Ticker` | Text | yes | Examples: `INFY.NS`, `TCS.BO`, `AAPL`, or `CASH` |
| `TransactionType` | Text | yes | `Buy`, `Sell`, `Deposit`, or `Withdraw` |
| `NoOfShares` | Number | yes | Share count, or `1` for cash rows |
| `CostPerShare` | Number | yes | Native price for stock rows; cash amount for `CASH` rows |
| `Date` | Date | yes | Format: `YYYY-MM-DD` |
| `Portfolio` | Text | optional | Defaults to `Default` when not supplied |
| `CMP` | Number | optional | Current market price patched by CMP updates |
| `Note` | Long text | optional | Thesis, trigger, risk, or rationale used by alerts/rebalance |

Stock buys and sells automatically create corresponding `CASH` rows. Manual deposits and withdrawals should also use `Ticker = CASH`.

### Table 4: PriceAlerts

| Field Name | Type | Required | Notes |
|---|---|---|---|
| `Id` | Auto-number | auto | NocoDB primary key |
| `Ticker` | Text | yes | Ticker to monitor |
| `Direction` | Text | yes | `above`, `below`, or `both` |
| `TargetPrice` | Number | yes | Alert trigger price |
| `Action` | Long text | optional | Optional instruction shown when alert fires |

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Telegram
TELEGRAM_TOKEN=your_bot_token_from_botfather
TELEGRAM_USER_ID=your_telegram_numeric_user_id
CHAT_ID=your_telegram_chat_id

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
TEXT_TO_SPEECH_MODEL=whisper-1

# NocoDB
NOCODB_BASE_URL=http://your-nocodb-host:8080
NOCODB_API_TOKEN=your_nocodb_api_token
NOCODB_EXPENSES_TABLE_ID=md_xxxxxxxx
NOCODB_MAG_TABLE_ID=md_yyyyyyyy
NOCODB_TRANSACTIONS_TABLE_ID=md_zzzzzzzz
NOCODB_PRICE_ALERTS_TABLE_ID=md_aaaaaaaa
NOCODB_EXPENSES_MAG_LINK_ID=link_field_id

# Wolfram Alpha
WOLFRAM_APP_ID=your_wolfram_app_id

# Wake/DDNS relay
PC_MAC_ADDRESS=AA:BB:CC:DD:EE:FF
BROADCAST_IP=192.168.1.255
RELAY_BASE_URL=http://172.17.0.1:9393
NOIP_USERNAME=your_noip_username
NOIP_PASSWORD=your_noip_password
NOIP_HOSTNAME=yourhost.ddns.net

# Optional rebalance tuning
REBALANCE_OPENAI_MODEL=gpt-5.4
REBALANCE_PRICE_INPUT_PER_1M=2.50
REBALANCE_PRICE_OUTPUT_PER_1M=15.00
REBALANCE_SCREEN_SIZE_PER_CATEGORY=30
```

`RELAY_BASE_URL` defaults to `http://172.17.0.1:9393` when omitted.

---

## Project Structure

```text
jarvis/
├── bujo/
│   ├── bujo-bot.py              # Executable wrapper for bujo.app.run()
│   ├── app.py                   # Telegram application, handlers, conversations, callbacks
│   ├── agent.py                 # Top-level LangGraph routing agent and analytics/Wolfram tools
│   ├── base.py                  # Env config, OpenAI clients, NocoDB models, auth decorator, scheduler
│   ├── scheduler.py             # APScheduler jobs
│   ├── handlers/                # Telegram handlers for chat, portfolio, alerts, and system commands
│   ├── expenses/manage.py       # ExpenseManager inner agent
│   ├── mag/manage.py            # MagManager inner agent
│   ├── portoflio/               # Portfolio manager, ledger, alerts, rebalance pipeline
│   ├── models/                  # NocoDB REST models
│   └── analytics/charts.py      # Matplotlib chart generation
├── tests/bujo/                  # Unit tests for MAG, analytics, portfolio ledger/alerts/rebalance
├── wake_relay.py                # Optional local relay for Wake-on-LAN and No-IP DDNS
├── Dockerfile                   # Python 3.13-slim image, non-root bot user
├── requirements.txt
├── pytest.ini
├── LICENSE
└── README.md
```

The `portoflio` directory name is intentionally documented as it exists on disk.

---

## Tech Stack

| Component | Technology |
|---|---|
| Bot framework | `python-telegram-bot` 22.7 |
| LLM and vision | OpenAI API |
| Agent framework | LangGraph ReAct agents with `MemorySaver` |
| Speech-to-text | OpenAI Whisper |
| Database | NocoDB REST API v2 |
| Charts | Matplotlib |
| Knowledge engine | Wolfram Alpha |
| Stock data | yfinance |
| Scheduling | APScheduler |
| Wake-on-LAN | Relay endpoint plus `wakeonlan` in `wake_relay.py` |
| Containerization | Docker, Python 3.13-slim |

---

## Getting Started

### Prerequisites

- Docker installed on the host
- A running NocoDB instance with the four tables above
- A Telegram bot token from BotFather
- An OpenAI API key
- A Wolfram Alpha App ID
- Optional: the relay service from `wake_relay.py` if you want Wake-on-LAN or DDNS commands

### 1. Clone And Configure

```bash
git clone https://github.com/snemmani/jarvis.git
cd jarvis
touch .env
```

Fill `.env` with the environment variables listed above.

### 2. Build The Docker Image

```bash
docker build -t jarvis .
```

### 3. Run The Bot

```bash
docker run -d \
  --network=host \
  --name jarvis \
  --env-file .env \
  jarvis
```

The container starts `python bujo/bujo-bot.py`, which delegates to `bujo.app.run()`.

### 4. Rebuild After Code Changes

```bash
docker rm -f jarvis
docker build -t jarvis .
docker run -d --network=host --name jarvis --env-file .env jarvis
```

---

## Commands

| Command | Description |
|---|---|
| `/start` | Greeting message |
| `/updateTicker` | Manually update CMP values for all portfolio tickers |
| `/portfolioDashboard` | Open the inline portfolio dashboard |
| `/portfolioAlerts` | Run portfolio risk/thesis alerts now |
| `/rebalanceRecommendations` | Generate a rebalance prompt document, ask for approval, then call OpenAI |
| `/buildPortfolio` | Start a guided fresh-portfolio planning conversation |
| `/setAlert <ticker> <above\|below\|both> <price> [action]` | Create a ticker price alert |
| `/listAlerts` | List active alerts and show modify/cancel buttons |
| `/ddns update` | Ask the relay to update No-IP hostname to current public IPv6 |
| `/ddns block` | Ask the relay to mangle the IPv6 target and block access |
| `/wakeTheBeast` | Ask the relay to send a Wake-on-LAN packet |
| `/genPass <length>` | Generate a secure random password |
| `/cancel` | Cancel the `/buildPortfolio` conversation |

Natural language messages do not need commands:

| What you say | What happens |
|---|---|
| `Spent 500 on groceries today` | Adds an expense and links it to today's MAG row |
| `Show me expenses for last week` | Lists filtered expenses |
| `Show me spending by category for April` | Sends a chart image |
| `Show me MAG for this week` | Lists calendar entries |
| `I completed my exercise today` | Updates today's MAG exercise field |
| `Bought 50 INFY.NS at 1500 today in LT portfolio because results were strong` | Adds stock transaction, note, and automatic cash ledger row |
| `Deposited 50000 cash into LT portfolio` | Adds a `CASH` deposit transaction |
| `Show my portfolio transactions for March` | Lists portfolio transactions with date filters |
| `Translate hello to Sanskrit` | Uses GPT translation |
| `What is the mass of the sun?` | Queries Wolfram Alpha and sends result pods |

---

## Implementation Details

### Agent Routing

Every text message, voice transcription, or image-parsed result flows into `agent_engage()` in `bujo/agent.py`. It builds a LangGraph ReAct agent with:

- `Expenses_Interaction`
- `MAG_interaction`
- `Portfolio_Tool`
- `Translation_Tool`
- `Wolfram_Alpha_Tool`
- `Expense_Analytics_Tool`

Wolfram and analytics tools are created per request because they need the active Telegram `update` and `context` to send images.

### Manager Sub-Agents

Expenses, MAG, and Portfolio each have their own inner ReAct agent. The top-level agent routes the request; the manager agent performs extraction, chooses the model tool call, and formats the response.

### Voice And Image Flow

Voice messages are downloaded as OGG files, transcribed with `TEXT_TO_SPEECH_MODEL`, and routed as text.

Photo messages are downloaded, base64 encoded, and sent to the configured OpenAI model. The model classifies the image as:

1. Expense/payment/receipt
2. Portfolio transaction/trade confirmation
3. Anything else

Captions have higher authority than the image for ticker, portfolio, and note overrides.

### NocoDB REST Client

`BaseNocoDB` centralizes NocoDB API calls. List operations use `_paginated_list()` so the bot fetches all pages instead of only the first page.

Filters use NocoDB's query syntax, for example:

```text
(Date,ge,exactDate,2026-05-01)
(Date,lt,exactDate,2026-06-01)
(Ticker,eq,text,INFY.NS)
```

### Portfolio Ledger And Dashboard

`PortfolioManager.add_transaction()` normalizes transaction fields, creates the stock row, and adds an offsetting `CASH` row for buys/sells. US-listed holdings are converted to INR for dashboard and P&L calculations.

`PortfolioManager.get_dashboard_data()` builds the inline dashboard data used by `/portfolioDashboard`: total value, equity value, cash, unrealized P&L, top holdings, per-portfolio summaries, and risk flags.

### Rebalance Flow

`/rebalanceRecommendations` prepares a full input document using holdings, cash ledger, transaction notes, forward-growth context, valuation outputs, headlines, and candidate screens. The bot sends that prompt to Telegram first, then only calls OpenAI after the user taps **Forward To LLM**.

The scheduled monthly rebalance job runs automatically on the first day of each month and sends both the input prompt and markdown report.

### Price Alerts

`/setAlert` creates rows in `PriceAlerts`. `/listAlerts` renders active alerts with inline buttons. If the user chooses modify, the next text message is interpreted as the new target price.

The scheduled price alert job runs during market hours and uses yfinance to compare CMP against alert targets.

### Scheduled Jobs

| Time (IST) | Days | Job |
|---|---|---|
| 08:00 | Daily | Send today's MAG entry to `CHAT_ID` |
| 08:15 | Mon-Fri | Update CMP for all portfolio tickers |
| 09:00 | Daily | Run portfolio alerts |
| 09:00, 11:00, 13:00, 15:00 | Mon-Fri | Check price alerts |
| 09:30 | Day 1 monthly | Run scheduled rebalance and send prompt/report files |

---

## Testing

Run the test suite from the repository root:

```bash
env PYTHONPATH=. venv/bin/pytest
```

The tests cover MAG model behavior, expense analytics, portfolio alerts, portfolio ledger calculations, portfolio manager behavior, and rebalance forward-context helpers.

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
