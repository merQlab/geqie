FROM python:3.12.8-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# geqie source
COPY geqie /geqie
RUN uv pip install /geqie --system

# app source
COPY gui /app
RUN uv pip install -r /app/requirements/requirements.txt --system

# default images
COPY assets/test_images /app/assets/test_images

RUN python manage.py collectstatic --noinput


FROM python:3.12.8-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 8000
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
