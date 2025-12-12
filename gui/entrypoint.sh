#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
DB_HOST="${POSTGRES_HOST:-db}"
DB_PORT="${POSTGRES_PORT:-5432}"
DEV_MODE="${DEV_MODE:-0}"

echo "===> Waiting for database ${DB_HOST}:${DB_PORT} ..."
for i in $(seq 1 60); do
  (echo > /dev/tcp/${DB_HOST}/${DB_PORT}) >/dev/null 2>&1 && break || true
  sleep 1
done

cd "$APP_DIR"

if [ -f "$APP_DIR/manage.py" ]; then
  MP="$APP_DIR/manage.py"
elif [ -f "$APP_DIR/gui/manage.py" ]; then
  MP="$APP_DIR/gui/manage.py"
else
  echo "❌ manage.py not found in /app or /app/gui"
  ls -la "$APP_DIR" || true
  ls -la "$APP_DIR/gui" || true
  exit 1
fi

if [ "${DJANGO_STRICT_MIGRATIONS:-0}" = "1" ]; then
  if ! python "$MP" makemigrations --check --dry-run; then
    echo "❌ Pending model changes detected. Commit migrations or unset DJANGO_STRICT_MIGRATIONS."
    exit 1
  fi
else
  python "$MP" makemigrations --noinput || true
fi

echo "===> Applying migrations..."
python "$MP" migrate --noinput

if [ "${DJANGO_COLLECTSTATIC:-1}" = "1" ]; then
  echo "===> Collect static..."
  python "$MP" collectstatic --noinput || true
fi

ENCODINGS_DIR="${ENCODINGS_DIR:-${APP_DIR}/assets/geqie/encodings}"
mkdir -p "$ENCODINGS_DIR" || true

echo "===> which geqie: $(command -v geqie || echo 'NOT FOUND')"
python - <<'PY'
try:
    import geqie, geqie.cli, shutil
    print("===> geqie module:", geqie.__file__)
    print("===> geqie bin   :", shutil.which("geqie"))
except Exception as e:
    print("===> geqie import problem:", e)
PY

WORKERS="${WEB_CONCURRENCY:-3}"
if [ "${DJANGO_ASGI:-0}" = "1" ]; then
  if python -c "import uvicorn" 2>/dev/null; then
    echo "===> Starting ASGI (gunicorn+uvicorn) on :8000 ..."
    exec gunicorn gui.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers "$WORKERS"
  else
    echo "!! uvicorn not installed, falling back to WSGI"
    exec gunicorn gui.wsgi:application --bind 0.0.0.0:8000 --workers "$WORKERS"
  fi
else
  echo "===> Starting WSGI (gunicorn) on :8000 ..."
  exec gunicorn gui.wsgi:application --bind 0.0.0.0:8000 --workers "$WORKERS"
fi
