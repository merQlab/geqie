import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gui.settings")

app = Celery("processing_queue")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()