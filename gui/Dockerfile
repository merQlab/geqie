# ./Dockerfile
FROM python:3.12.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . .

RUN pip install -e /app

RUN cp gui/entrypoint.sh /app/entrypoint.sh \
 && sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

RUN python gui/manage.py collectstatic --noinput || true

EXPOSE 8000

CMD ["/app/entrypoint.sh"]