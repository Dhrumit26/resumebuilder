FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most PaaS) inject PORT at runtime — often 10000.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} expands. exec replaces the shell so signals reach uvicorn.
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
