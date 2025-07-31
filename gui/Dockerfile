# Use an official Python runtime as a parent image
FROM python:3.12.8-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Build statics
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "gui.wsgi:application", "--bind", "0.0.0.0:8000"]
