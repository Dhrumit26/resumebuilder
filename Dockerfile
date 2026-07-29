FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud hosts inject PORT; default 8000 for local docker runs
ENV PORT=8000
EXPOSE 8000

# Long LLM builds need a high timeout — do not use a serverless 10s platform
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
