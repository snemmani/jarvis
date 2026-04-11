<p align="center">
  <h1 align="center">🤖 JARVIS Bot</h1>
  <p align="center">
    <strong>Your AI-powered personal assistant on Telegram</strong>
  </p>
  <p align="center">
    Manage expenses · Track your calendar · Monitor your portfolio · And more
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#commands">Commands</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## ✨ Overview

JARVIS Bot is a personal Telegram bot that acts as an intelligent assistant for managing daily finances, a personal calendar (called **MAG** — Month/Day at a Glance), and an investment portfolio. It uses **OpenAI GPT** via **LangChain/LangGraph** to understand natural language, route requests to the right tools, and respond in a friendly, emoji-rich markdown format.

Send it a text message, a voice note, or even a photo of a receipt — JARVIS figures out what you need and gets it done.

---

## 🚀 Features

### 💸 Expense Management
- **Add expenses** in natural language — *"Spent 500 on groceries today"*
- **Send a photo** of a receipt or payment screenshot and JARVIS will extract the amount, item, and date using GPT vision
- **List & filter expenses** by date range, week, month, or specific day
- Expenses are automatically linked to the corresponding MAG (calendar) entry
- All amounts displayed in **₹ (Indian Rupees)**

### 📅 MAG — Personal Calendar
- **View your calendar** — *"Show me MAG for this week"*
- **Update notes & exercise tracking** — *"I completed my exercise today"*
- Tracks: Date, Day, Tithi (Telugu calendar), Notes, Exercise status, and daily expense totals
- **Scheduled daily briefing** — Automatically sends today's MAG at **8:00 AM IST**

### 📊 Portfolio Tracker
- Track **Indian (NSE)** and **US stock** positions
- **Live CMP updates** via [yfinance](https://github.com/ranaroussi/yfinance) — scheduled every weekday at **8:15 AM IST**
- **Profit & Loss reports** with unrealised/realised P&L, grouped by portfolio
- Automatic **USD → INR** conversion for US-listed stocks
- Commands: `/updateTicker`, `/getProfitLoss`

### 🧠 AI-Powered Intelligence
- **Natural language understanding** — no rigid command syntax needed
- **Voice messages** — speak to JARVIS and it transcribes + processes via OpenAI Whisper
- **Image analysis** — send photos of transactions/receipts for automatic expense logging
- **Wolfram Alpha integration** — ask about math, science, conversions, nutrition, stock prices, flight fares, and more
- **Translation** — translate text between languages (defaults to English → Sanskrit)
- **Conversational memory** — maintains context across messages per user session

### 🔧 Utility Commands
- `/wakeTheBeast` — Send a Wake-on-LAN magic packet to power on your PC remotely
- `/genPass <length>` — Generate a cryptographically secure random password

### 🔒 Security
- **User authorization** — only whitelisted Telegram user IDs can interact with the bot
- Unauthorized messages are forwarded to the owner for awareness

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Telegram User                      │
│          (Text / Voice / Photo / Commands)            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              bujo-bot.py (Main Bot)                   │
│  ┌─────────────────────────────────────────────────┐ │
│  │         LangGraph ReAct Agent (GPT)             │ │
│  │  ┌───────────┬───────────┬───────────┬────────┐ │ │
│  │  │ Expenses  │    MAG    │  Wolfram  │ Transl.│ │ │
│  │  │   Tool    │   Tool    │   Alpha   │  Tool  │ │ │
│  │  └─────┬─────┴─────┬─────┴─────┬─────┴────────┘ │ │
│  └────────┼───────────┼───────────┼────────────────┘ │
│           ▼           ▼           ▼                   │
│  ┌──────────────────────────────────────┐            │
│  │        Manager Layer                  │            │
│  │  ExpenseManager │ MagManager │        │            │
│  │  PortfolioManager                     │            │
│  │  (Each with its own LangGraph agent)  │            │
│  └──────────────────┬───────────────────┘            │
│                     ▼                                 │
│  ┌──────────────────────────────────────┐            │
│  │        Model Layer (NocoDB)           │            │
│  │  Expenses │ MAG │ PortfolioTransact.  │            │
│  │  (BaseNocoDB → REST API)              │            │
│  └──────────────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     NocoDB       │
              │  (Database API)  │
              └─────────────────┘
```

### Key Design Decisions

- **Multi-agent architecture** — A top-level ReAct agent routes to specialized sub-agents (Expenses, MAG, Portfolio), each with their own system prompts and tool sets
- **NocoDB as backend** — All data (expenses, calendar, portfolio) is stored in NocoDB tables via its REST API, making it easy to view/edit data outside the bot
- **LangGraph with memory** — Conversational state is maintained per user via `MemorySaver` checkpointers
- **APScheduler** — Cron-based scheduling for daily MAG briefings and weekday CMP updates

---

## 📁 Project Structure

```
expenses-bot/
├── bujo/
│   ├── bujo-bot.py              # Main entry point — Telegram handlers & top-level agent
│   ├── base.py                  # Configuration, env vars, LLM init, auth decorator
│   ├── expenses/
│   │   └── manage.py            # ExpenseManager — sub-agent for expense CRUD
│   ├── mag/
│   │   └── manage.py            # MagManager — sub-agent for calendar management
│   ├── portoflio/
│   │   └── manage.py            # PortfolioManager — portfolio tracking & P&L reports
│   └── models/
│       ├── base.py              # BaseNocoDB — generic REST client for NocoDB
│       ├── expenses.py          # Expenses model — CRUD + MAG linking
│       ├── mag.py               # MAG model — calendar CRUD + date lookup
│       └── portfolio_transactions.py  # Portfolio model — transaction CRUD + CMP updates
├── tests/                       # Test directory
├── Dockerfile                   # Docker containerization (Python 3.13-slim)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not committed)
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| **Bot Framework** | [python-telegram-bot](https://python-telegram-bot.org/) |
| **LLM** | [OpenAI GPT](https://openai.com/) via [LangChain](https://www.langchain.com/) |
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) (ReAct agents) |
| **Speech-to-Text** | OpenAI Whisper |
| **Vision** | OpenAI GPT Vision |
| **Database** | [NocoDB](https://nocodb.com/) (REST API) |
| **Knowledge Engine** | [Wolfram Alpha](https://www.wolframalpha.com/) |
| **Stock Data** | [yfinance](https://github.com/ranaroussi/yfinance) |
| **Scheduling** | [APScheduler](https://apscheduler.readthedocs.io/) |
| **Wake-on-LAN** | [wakeonlan](https://pypi.org/project/wakeonlan/) |
| **Containerization** | Docker (Python 3.13-slim) |

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ as a personal productivity companion
</p>
