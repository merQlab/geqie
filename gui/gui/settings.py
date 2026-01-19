from pathlib import Path
from botocore.config import Config
import os
import geqie

BASE_DIR = Path(__file__).resolve().parent.parent

os.makedirs(BASE_DIR / "logs", exist_ok=True)

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-change-me"
)

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

ALLOWED_HOSTS = os.environ.get(
    "ALLOWED_HOSTS",
    "localhost,127.0.0.1"
).split(",")
CSRF_TRUSTED_ORIGINS = [origin for origin in os.environ.get("CSRF_TRUSTED_ORIGINS", "").split(",") if origin]

# --- APPLICATIONS -------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "crispy_forms",
    "crispy_bootstrap5",
    "channels",
    "storages",
    "main",
]

# --- MIDDLEWARE / URL / WSGI/ASGI -----------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "gui.urls"
WSGI_APPLICATION = "gui.wsgi.application"
ASGI_APPLICATION = "gui.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# --- CHANNELS (Redis) ------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [os.environ.get("REDIS_URL", "redis://redis:6379/0")],
        },
    },
}

# --- DATABASE -----------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": os.environ.get("DB_ENGINE", "django.db.backends.postgresql"),
        "NAME": os.environ.get("POSTGRES_DB", "geqie"),
        "USER": os.environ.get("POSTGRES_USER", "geqie"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "HOST": os.environ.get("POSTGRES_HOST", os.environ.get("DB_HOST", "db")),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

# --- AUTHORIZATION ----------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- LOCALIZATION -----------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("TZ", "Europe/Warsaw")
USE_I18N = True
USE_TZ = True

# --- STATIC (Whitenoise + MinIO/S3) ---------------------------------------
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "main" / "static" / "geqie"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# --- MEDIA (MinIO/S3) -----------------------------------------------------
STORAGES = {
    "default": {
        "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}
AWS_S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
AWS_STORAGE_BUCKET_NAME = os.environ.get("S3_BUCKET", "media")
AWS_S3_REGION_NAME = os.environ.get("S3_REGION", "us-east-1")
AWS_S3_SIGNATURE_VERSION = "s3v4"
AWS_S3_ADDRESSING_STYLE = "path"
AWS_QUERYSTRING_AUTH = True
AWS_DEFAULT_ACL = None
AWS_S3_PUBLIC_ENDPOINT_URL = os.environ.get("S3_PUBLIC_ENDPOINT", AWS_S3_ENDPOINT_URL)
BOTO_S3_CONFIG = Config(s3={"addressing_style": "path"}, signature_version="s3v4")

MEDIA_URL = "/media/"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            "datefmt": "%d/%b/%Y %H:%M:%S",
        },
        "simple": {"format": "%(levelname)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "verbose",
            "level": "ERROR",
            "filename": os.path.join(BASE_DIR, "logs", "error.log"),
            "when": "midnight",
            "interval": 30,
            "backupCount": 2,
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "DEBUG" if DEBUG else "INFO",
            "propagate": True,
        },
        "main": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "frontend_logs": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- CELERY ----------------------------------------------------------------
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_TASK_DEFAULT_QUEUE = "processing_queue"
CELERY_TASK_ROUTES = {
    "main.tasks.run_experiment": {"queue": "processing_queue"},
}

CRISPY_TEMPLATE_PACK = "bootstrap5"

DEFAULT_INIT = {
    "__init__.py": (
        "from .init import init as init_function\n"
        "from .data import data as data_function\n"
        "from .map import map as map_function\n"
        "from .retrieve import retrieve as retrieve_function"
    ),
}

DEFAULT_METHODS_CONTENT = {
    "init": """import numpy as np
from qiskit.quantum_info import Statevector

def init(n_qubits: int) -> Statevector:
    qubits_in_superposition = # place number of qubits in superposition
    base_state = np.zeros(2**qubits_in_superposition, dtype=int)
    base_state[0] = 1
    state = np.tile(base_state, 2**(n_qubits - qubits_in_superposition))
    return Statevector(state)""",
    "map": """import numpy as np
from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    p = image[u, v]
    # Provide your own unitary matrix for map operator
    return Operator(map_operator)""",
    "data": """import numpy as np
from qiskit.quantum_info import Statevector

def data(u: int, v: int, R: int, image: np.ndarray) -> Statevector:
    m = u * image.shape[0] + v
    data_vector = np.zeros(2**(2 * R))
    data_vector[m] = 1
    return Statevector(data_vector)""",
    "retrieve": """import numpy as np
import json

def retrieve(results: str) -> np.ndarray:
    \"""
    Decodes an image from quantum state measurement results.
    \"""
    state_length = len(next(iter(results)))
    # color_qubits = set qubits used for color encoding
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    image_shape = (2**x_qubits, 2**y_qubits)

    # Provide your own code here...
    return reconstructed_image""",
}

ENCODINGS_DIR = (Path(geqie.__file__).resolve().parent / "encodings").resolve()