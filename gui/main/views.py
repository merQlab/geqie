import os
import json
import uuid
import base64
import logging
import mimetypes
from main.services.method_approval import require_approved_method
from pathlib import Path

from django.conf import settings
from django.core import signing
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse, Http404, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from .models import Job, QuantumMethod, QuantumComputer
from .tasks import run_experiment
from .utils import all_methods, approved_methods, refresh_quantum_methods

logger = logging.getLogger(__name__)
loggerFront = logging.getLogger("frontend_logs")

# Temporary proxy TTL (seconds)
PROXY_URL_TTL = getattr(settings, "PROXY_URL_TTL", 300)


def home(request):
    return render(request, "home.html")


def experiment_config(request):
    methods = approved_methods()
    computers = QuantumComputer.objects.all()
    return render(request, "experiment_config.html", {"methods": methods, "computers": computers})


def edit_method(request):
    methods = all_methods()
    default_methods_content = settings.DEFAULT_METHODS_CONTENT
    logger.debug("Edit_method card")
    return render(
        request,
        "edit_method.html",
        {"methods": methods, "default_methods_content": default_methods_content},
    )


def _s3_client(endpoint_url: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=getattr(settings, "AWS_S3_REGION_NAME", None),
        config=getattr(settings, "BOTO_S3_CONFIG", None),
    )


def _proxy_url(request, job_id: str, key: str, content_type: str | None = None) -> str | None:
    """
    Returns a temporary Django-hosted URL that streams the object from S3/MinIO.
    The token is signed and expires via max_age in proxy_file().
    """
    if not key:
        return None
    token = signing.dumps({"job_id": str(job_id), "key": key, "content_type": content_type})
    return request.build_absolute_uri(f"/media/proxy/{token}/")


@require_GET
def proxy_file(request, token: str):
    """
    Streams an object from S3/MinIO through Django using a short-lived signed token.
    """
    try:
        data = signing.loads(token, max_age=PROXY_URL_TTL)
    except signing.BadSignature:
        raise Http404("Invalid or expired link")

    job_id = data.get("job_id")
    key = data.get("key")
    if not job_id or not key:
        raise Http404("Invalid token payload")

    # Fetch job and ensure the key belongs to this job
    try:
        job = Job.objects.get(pk=job_id)
    except Job.DoesNotExist:
        raise Http404("Job not found")

    allowed_keys = {
        job.input_key,
        job.output_json_key,
        job.original_png_key,
        job.retrieved_png_key,
    }
    if key not in allowed_keys:
        raise Http404("Not allowed")

    content_type = data.get("content_type")
    if not content_type:
        guessed, _ = mimetypes.guess_type(key)
        content_type = guessed or "application/octet-stream"

    s3 = _s3_client(settings.AWS_S3_ENDPOINT_URL)

    try:
        obj = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=key)
    except Exception:
        raise Http404("File not found")

    body = obj["Body"]
    resp = FileResponse(body, content_type=content_type)

    content_length = obj.get("ContentLength")
    if content_length is not None:
        resp["Content-Length"] = str(content_length)

    etag = obj.get("ETag")
    if etag:
        resp["ETag"] = etag

    resp["Cache-Control"] = "no-store"
    return resp


@require_POST
def start_experiment(request):
    logger.info("Processing POST /start-experiment/")

    selected_method = request.POST.get("selected_method")
    shots = request.POST.get("shots", "1024")
    is_retrieve = request.POST.get("is_retrieve", "").strip().lower() in ["true"]

    if not selected_method:
        return JsonResponse({"success": False, "error": "No method selected."}, status=400)

    # APPROVAL GATE (single source of truth, fail closed)
    require_approved_method(selected_method)

    file_list = request.FILES.getlist("images[]") or list(request.FILES.values())

    if not file_list:
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}

        assets_root = Path("/app/assets")
        roots = [
            assets_root / "test_images" / "grayscale",
            assets_root / "test_images" / "rgb",
        ]

        gathered = []
        for root in roots:
            if root.exists():
                for p in root.rglob("*"):
                    if p.is_file() and p.suffix.lower() in exts:
                        gathered.append(p)

        if not gathered:
            searched = ", ".join(str(p) for p in roots)
            return JsonResponse(
                {"success": False, "error": f"No files provided and no images found under: {searched}"},
                status=400
            )

        for p in gathered:
            data = p.read_bytes()
            key = f"uploads/{uuid.uuid4()}_{p.name}"
            default_storage.save(key, ContentFile(data))
            file_list.append({"_stored_key": key, "_orig_name": p.name})

    jobs = []
    for uploaded in file_list:
        if hasattr(uploaded, "name"):
            key = f"uploads/{uuid.uuid4()}_{uploaded.name}"
            default_storage.save(key, uploaded)
            filename = uploaded.name
        else:
            key = uploaded["_stored_key"]
            filename = uploaded["_orig_name"]

        job = Job.objects.create(
            filename=filename,
            method=selected_method,
            shots=str(shots),
            is_retrieve=is_retrieve,
            input_key=key,
            status="queued",
        )
        run_experiment.delay(str(job.id))
        jobs.append({"file": filename, "job_id": str(job.id)})

    logger.info("Queued %d job(s) for method=%s shots=%s", len(jobs), selected_method, shots)
    return JsonResponse({"jobs": jobs})



@require_GET
def get_config(request):
    """Return configuration values needed by the frontend."""
    return JsonResponse({
        "max_image_size": settings.MAX_IMAGE_SIZE,
    })


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
                "output_json_url": _proxy_url(request, job.id, job.output_json_key, "application/json"),
                "original_url": _proxy_url(request, job.id, job.original_png_key, "image/png"),
                "retrieved_url": _proxy_url(request, job.id, job.retrieved_png_key, "image/png"),
            }
        )
    return JsonResponse(payload)


def read_method_files(request, method_name):
    method_path = os.path.join(settings.ENCODINGS_DIR, method_name)
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

        method_path = os.path.join(settings.ENCODINGS_DIR, method_name)

        if is_new and not os.path.exists(os.path.join(settings.ENCODINGS_DIR, save_name)):
            method_path = os.path.join(settings.ENCODINGS_DIR, save_name)
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


@require_GET
def get_all_images(request):
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
    images = []

    try:
        assets_root = Path("/app/assets")
        roots = [
            assets_root / "test_images" / "grayscale",
            assets_root / "test_images" / "rgb",
        ]

        any_found = False
        for root in roots:
            if not root.exists():
                logger.debug("get_all_images: skipping the missing directory: %s", root)
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    body = p.read_bytes()
                    mime_type, _ = mimetypes.guess_type(p.name)
                    images.append({
                        "name": p.name,
                        "data": base64.b64encode(body).decode("utf-8"),
                        "type": mime_type or "application/octet-stream",
                    })
                    any_found = True

        if not any_found:
            logger.warning("get_all_images: no images found. Searched in: %s", ", ".join(map(str, roots)))
        return JsonResponse({"images": images})
    except Exception as e:
        logger.exception("get_all_images error: %s", e)
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
