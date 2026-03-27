FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[hpo,lightgbm]"
COPY configs ./configs

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

ENV CREDIT_RISK_CONFIG=/app/configs/pipeline.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "credit_risk.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
