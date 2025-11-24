# ./Dockerfile
FROM python:3.12.8-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for building Python packages and PostgreSQL support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY gui/requirements/requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt --system

COPY . .

RUN uv pip install -e /app --system

RUN cp gui/entrypoint.sh /app/entrypoint.sh \
    && sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

RUN python gui/manage.py collectstatic --noinput || true

EXPOSE 8000

CMD ["/app/entrypoint.sh"]