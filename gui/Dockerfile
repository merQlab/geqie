# syntax=docker/dockerfile:1.7

FROM python:3.12.8-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY gui/requirements/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/requirements.txt --system

# app source
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /app --system

RUN cp gui/entrypoint.sh /app/entrypoint.sh \
    && sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

RUN python gui/manage.py collectstatic --noinput


FROM python:3.12.8-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 8000
CMD ["/app/entrypoint.sh"]
