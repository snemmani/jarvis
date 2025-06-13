FROM python:3.13-slim

ENV TZ="Asia/Kolkata"
# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Set additional environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"

# Command to run the bot
CMD ["python", "bujo/bujo-bot.py"]
