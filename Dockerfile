≈FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.pip.txt /app/requirements.pip.txt
RUN pip install --no-cache-dir -r requirements.pip.txt

COPY . /app

CMD ["sh", "-c", "uvicorn triage_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
