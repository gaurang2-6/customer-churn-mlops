# Serving image: only the API, the trained model and its runtime dependencies (no MLflow, no training code deps).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

# libgomp is needed by XGBoost on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY churn/__init__.py churn/api.py churn/model.py churn/data.py churn/
COPY models/ models/

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port ${PORT}"]
