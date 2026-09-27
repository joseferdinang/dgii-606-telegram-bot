FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DATA_DIR=/app/data \
    STATE_DB_PATH=/app/data/state/bot606.sqlite3 \
    EXPORT_DIRECTORY=/app/data/exports

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 bot606 \
    && useradd --system --uid 10001 --gid bot606 --create-home bot606

COPY requirements.txt ./
RUN pip install --requirement requirements.txt

COPY --chown=bot606:bot606 . ./
RUN mkdir -p /app/data/state /app/data/exports \
    && chown -R bot606:bot606 /app

USER bot606

CMD ["python", "main.py"]
