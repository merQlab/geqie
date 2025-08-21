import io
import os
import json
import uuid
import base64
import logging
from importlib import import_module

from django.conf import settings
from django.http import JsonResponse, Http404
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.core.files.storage import default_storage

from .models import QuantumMethod, QuantumComputer, Job
from .utils import all_methods, approved_methods, refresh_quantum_methods
from .tasks import run_experiment
from gui.settings import ENCODINGS_DIR

logger = logging.getLogger(__name__)
loggerFront = logging.getLogger("frontend_logs")


def home(request):
    return render(request, "home.html")


def experiment_config(request):
    methods = approved_methods()
    computers = QuantumComputer.objects.all()
    logger.info("Experiment_config card")
    return render(request, "experiment_config.html", {"methods": methods, "computers": computers})


def edit_method(request):
    methods = all_methods()
    default_methods_content = settings.DEFAULT_METHODS_CONTENT
    logger.info("Edit_method card")
    return render(
        request,
        "edit_method.html",
        {"methods": methods, "default_methods_content": default_methods_content},
    )


def _s3_client(endpoint_url: str):
    import boto3
    from django.conf import settings
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=getattr(settings, "AWS_S3_REGION_NAME", None),
        config=getattr(settings, "BOTO_S3_CONFIG", None),
    )


def _presigned_url(key: str, ttl: int = 3600) -> str | None:
    if not key:
        return None
    s3_public = _s3_client(getattr(settings, "AWS_S3_PUBLIC_ENDPOINT_URL", settings.AWS_S3_ENDPOINT_URL))
    return s3_public.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.AWS_STORAGE_BUCKET_NAME, "Key": key},
        ExpiresIn=ttl,
    )


@csrf_exempt
@require_POST
def start_experiment(request):
    """
    It accepts files, saves to S3/MinIO, creates a Job and starts the Celery worker.
    Returns a list of job_ids for uploaded files.
    """
    logger.info("Processing POST /start-Experiment/")
    selected_method = request.POST.get("selected_method")
    shots = request.POST.get("shots", "1024")
    is_retrieve = request.POST.get("is_retrieve", "").strip().lower() in ["true"]

    file_list = request.FILES.getlist("images[]") or list(request.FILES.values())
    if not file_list:
        return JsonResponse({"success": False, "error": "No files provided."}, status=400)

    if not selected_method:
        return JsonResponse({"success": False, "error": "No method selected."}, status=400)

    jobs = []
    for uploaded in file_list:
        key = f"uploads/{uuid.uuid4()}_{uploaded.name}"
        default_storage.save(key, uploaded)

        job = Job.objects.create(
            filename=uploaded.name,
            method=selected_method,
            shots=str(shots),
            is_retrieve=is_retrieve,
            input_key=key,
            status="queued",
        )

        run_experiment.delay(str(job.id))
        jobs.append({"file": uploaded.name, "job_id": str(job.id)})

    logger.info("Queued %d job(s) for method=%s shots=%s", len(jobs), selected_method, shots)
    return JsonResponse({"jobs": jobs})


@require_GET
def job_status(request, job_id):
    try:
        job = Job.objects.get(pk=job_id)
    except Job.DoesNotExist:
        raise Http404("Job not found")

    payload = {
        "job_id": str(job.id),
        "file": job.filename,
        "status": job.status,
        "error": job.error,
    }
    if job.status == "done":
        payload.update(
            {
                "output_json_url": _presigned_url(job.output_json_key),
                "original_url": _presigned_url(job.original_png_key),
                "retrieved_url": _presigned_url(job.retrieved_png_key),
            }
        )
    return JsonResponse(payload)


def read_method_files(request, method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    if not os.path.exists(method_path):
        logger.error("Method not found: %s", method_name)
        return JsonResponse({"error": "Method not found"}, status=404)

    files = ["init.py", "map.py", "data.py", "retrieve.py"]
    file_contents = {}

    for file in files:
        file_path = os.path.join(method_path, file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                file_contents[file.replace(".py", "")] = f.read()
        else:
            file_contents[file.replace(".py", "")] = "File not found."

    return JsonResponse(file_contents)


@csrf_exempt
def save_method_files(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        data = json.loads(request.body)
        method_name = data.get("method_name")
        init_content = data.get("init")
        map_content = data.get("map")
        data_content = data.get("data")
        retrieve_content = data.get("retrieve")
        is_new = data.get("is_new", False)
        save_name = data.get("save_name")

        if not method_name:
            logger.error("No method name provided")
            return JsonResponse({"error": "No method name provided"}, status=400)

        method_path = os.path.join(ENCODINGS_DIR, method_name)

        if is_new and not os.path.exists(os.path.join(ENCODINGS_DIR, save_name)):
            method_path = os.path.join(ENCODINGS_DIR, save_name)
            os.makedirs(method_path)

        files = settings.DEFAULT_INIT.copy()
        files.update(
            {
                "init.py": init_content,
                "map.py": map_content,
                "data.py": data_content,
                "retrieve.py": retrieve_content,
            }
        )

        for filename, content in files.items():
            file_path = os.path.join(method_path, filename)
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            os.chmod(file_path, 0o755)

        refresh_quantum_methods()
        logger.info("Method files saved successfully in %s", method_path)
        return JsonResponse({"message": "Method saved successfully"})

    except Exception as e:
        logger.exception("Error writing method files: %s", str(e))
        return JsonResponse({"error": str(e)}, status=500)


def check_folder_exists(request):
    folder_name = request.GET.get("folder_name", "")
    folder_path = os.path.join(settings.ENCODINGS_DIR, folder_name)

    exists = os.path.exists(folder_path) and os.path.isdir(folder_path)
    return JsonResponse({"exists": exists})


def get_all_images(request):
    prefix = "grayscale/"
    images = []
    try:
        s3 = _s3_client()
        bucket = settings.AWS_STORAGE_BUCKET_NAME
        paginator = s3.get_paginator("list_objects_v2")

        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    keys.append(key)

        for key in keys:
            with default_storage.open(key, "rb") as f:
                data = f.read()
            name = os.path.basename(key)
            if name.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif name.lower().endswith(".gif"):
                mime_type = "image/gif"
            else:
                mime_type = "image/png"

            images.append({
                "name": name,
                "data": base64.b64encode(data).decode("utf-8"),
                "type": mime_type
            })

        return JsonResponse({"images": images})
    except Exception as e:
        logger.exception("S3 read error: %s", e)
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def log_from_js(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=405)

    try:
        data = json.loads(request.body)
        level = data.get("level", "info")
        message = data.get("message", "")

        if level == "debug":
            loggerFront.debug(message)
        elif level == "info":
            loggerFront.info(message)
        elif level == "warning":
            loggerFront.warning(message)
        elif level == "error":
            loggerFront.error(message)
        elif level == "critical":
            loggerFront.critical(message)
        else:
            loggerFront.info(f"[Unknown level] {message}")

        return JsonResponse({"status": "success"})
    except Exception as e:
        loggerFront.exception(f"Failed to process JS log: {str(e)}")
        return JsonResponse({"status": "error"}, status=400)