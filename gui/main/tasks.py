from __future__ import annotations
import logging
import io, os, sys, json, tempfile, subprocess, shutil
from main.services.method_approval import require_approved_method
from pathlib import Path
from collections import OrderedDict

from PIL import Image, ImageOps
try:
    import numpy as np
except Exception:
    np = None

from celery import shared_task
from django.conf import settings
from django.core.files.storage import default_storage
from importlib.util import spec_from_file_location, module_from_spec
from .models import Job

logger = logging.getLogger(__name__)
DEFAULT_JOB_TIMEOUT_SECONDS = 300  # 5 minutes
JOB_TIMEOUT_SECONDS = os.getenv("JOB_TIMEOUT_SECONDS", DEFAULT_JOB_TIMEOUT_SECONDS)


class UserVisibleError(Exception):
    pass

def _save_pil_to_s3(img: Image.Image, key: str):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    default_storage.save(key, buf)
    buf.close()


def _load_method_module(method_name: str):
    enc_dir = Path(settings.ENCODINGS_DIR)
    init_py = enc_dir / method_name
    if not init_py.exists():
        return None
    spec = spec_from_file_location(f"{method_name}", init_py)
    if not spec or not spec.loader:
        return None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _try_geqie_cli_simulate(method: str, image_path: str, shots: int) -> tuple[bool, OrderedDict | None, str]:
    geqie_bin = shutil.which("geqie")
    if geqie_bin:
        cmd = [
                geqie_bin, "simulate",
                "--encoding", str(method),
                "--image-path", image_path,
                "--n-shots", str(shots),
                "--return-padded-counts", "true"
            ]
    else:
        cmd = [sys.executable, "-m", "geqie.cli", "simulate",
               "--encoding", str(method),
               "--image-path", image_path,
               "--n-shots", str(shots),
               "--return-padded-counts", "true",
               "--verbosity-level", "INFO",
            ]

    try:
        proc = subprocess.run(args=cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

        return (True, OrderedDict(json.loads(proc.stdout)), "")
    except subprocess.TimeoutExpired:
        return (False, None, f"Timeout ({JOB_TIMEOUT_SECONDS} seconds) running the job.")
    except json.JSONDecodeError as e:
        return (False, None, f"Invalid JSON from geqie.simulate(): {e}")
    except subprocess.CalledProcessError as e:
        if proc.returncode == -9:  # KILLED
            return (False, None, f"geqie.simulate() process was killed (possible out-of-memory). Please consider using a smaller image.")
        else:
            return (False, None, f"geqie.simulate() process returned non-zero status code: {proc.returncode}, Error message: {e.stderr}")
    except Exception as e:
        return (False, None, f"Unexpected error running geqie.simulate(): {str(e)}")


def _to_pil(rec) -> Image.Image | None:
    if isinstance(rec, Image.Image):
        return rec.convert("RGB")
    if np is not None:
        try:
            arr = np.asarray(rec)
            if arr.dtype != 'uint8':
                arr = arr.astype('float32')
                arr = arr - arr.min()
                mx = arr.max() or 1.0
                arr = (arr / mx * 255.0).clip(0, 255).astype('uint8')
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                mode = "RGB" if arr.shape[2] == 3 else "RGBA"
                return Image.fromarray(arr, mode=mode).convert("RGB")
        except Exception:
            return None
    return None


@shared_task(
    bind=True, 
    queue="processing_queue",
    soft_time_limit=settings.CELERY_EXPERIMENT_SOFT_TIME_LIMIT,
    time_limit=settings.CELERY_EXPERIMENT_HARD_TIME_LIMIT,
)
def run_experiment(self, job_id: str) -> dict:
    logger.debug(f"Starting run_experiment for job_id: '{job_id}'")

    job = Job.objects.get(pk=job_id)
    require_approved_method(str(job.method))
    job.status = "running"
    job.error = ""
    job.save(update_fields=["status", "error", "updated_at"])

    ext = os.path.splitext(job.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_in:
        tmp_path = f_in.name
        with default_storage.open(job.input_key, "rb") as src:
            f_in.write(src.read())

    try:
        shots = int(job.shots) if str(job.shots).isdigit() else 1024

        ok, ordered_output, err = _try_geqie_cli_simulate(str(job.method), tmp_path, shots)
        if not ok:
            logger.error("geqie simulate failed for method=%s file=%s: %s", job.method, job.filename, err)
            raise UserVisibleError(f"Experiment failed: '{str(err)}'")

        out_json_key = f"results/{job.id}_output.json"
        payload = ordered_output.get("counts", ordered_output)
        default_storage.save(out_json_key, io.BytesIO(json.dumps(payload, separators=(",", ":")).encode("utf-8")))

        orig_key = f"results/{job.id}_original.png"
        _save_pil_to_s3(Image.open(tmp_path).convert("RGB"), orig_key)

        retrieved_png_key = None
        retrieved_img = None

        if getattr(job, "is_retrieve", False):
            mod = _load_method_module(str(job.method))
            if mod and hasattr(mod, "retrieve_function"):
                try:
                    rec = mod.retrieve_function(ordered_output)
                    retrieved_img = _to_pil(rec)
                except Exception as e:
                    job.error = (job.error + "\n" if job.error else "") + f"retrieve_function error: {e}"
            if retrieved_img is None:
                retrieved_img = ImageOps.invert(Image.open(tmp_path).convert("RGB"))

        if retrieved_img is not None:
            retrieved_png_key = f"results/{job.id}_retrieved.png"
            _save_pil_to_s3(retrieved_img, retrieved_png_key)

        if err:
            job.error = (job.error + "\n" if job.error else "") + err[:3500]

        job.output_json_key = out_json_key
        job.original_png_key = orig_key
        job.retrieved_png_key = retrieved_png_key
        job.status = "done"
        job.save(update_fields=[
            "output_json_key", "original_png_key", "retrieved_png_key", "status", "error", "updated_at"
        ])
        logger.debug(f"Job with job_id: '{job_id}' finished successfully")
        return {"ok": True}

    except UserVisibleError as e:
        job.status = "error"
        job.error = str(e)
        job.save(update_fields=["status", "error", "updated_at"])
        return {"ok": False, "error": job.error}
    except Exception as e:
        logger.exception("Unexpected error in run_experiment job_id=%s", job.id)
        job.status = "error"
        job.error = "Experiment failed: internal error: " + str(e)
        job.save(update_fields=["status", "error", "updated_at"])
        return {"ok": False,"error": job.error}