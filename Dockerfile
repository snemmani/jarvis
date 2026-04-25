FROM python:3.13-slim

ENV TZ="Asia/Kolkata"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN grep -v 'assert resp' /usr/local/lib/python3.13/site-packages/wolframalpha/__init__.py > /tmp/__init__.py && \
    mv /tmp/__init__.py /usr/local/lib/python3.13/site-packages/wolframalpha/__init__.py

# Install Node.js + Claude Code CLI (needed for /claudeApi command)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -s /bin/bash bot && chown -R bot:bot /app

ENV PYTHONPATH="/app:$PYTHONPATH"
USER bot

CMD ["python", "bujo/bujo-bot.py"]
