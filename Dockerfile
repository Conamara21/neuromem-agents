FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /app/
COPY neuromem /app/neuromem

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[server,mcp]"

EXPOSE 8080 8765

CMD ["neuromem-openai-server", "--host", "0.0.0.0", "--port", "8080"]
