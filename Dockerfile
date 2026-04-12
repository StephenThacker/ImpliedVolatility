FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY initialize_database.py .
COPY nightly_update.py .

COPY cronjob /etc/cron.d/cronjob
RUN chmod 0644 /etc/cron.d/cronjob && \
    crontab /etc/cron.d/cronjob

# Run cron in foreground so the container stays alive
CMD ["cron", "-f"]