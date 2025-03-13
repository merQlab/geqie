import io
from django.shortcuts import render
from django.http import JsonResponse
from .utils import all_methods, approved_methods
from gui.settings import ENCODINGS_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from .utils import refresh_quantum_methods
import matplotlib.pyplot as plt
import json
import subprocess
import os
import base64
import uuid
import logging
from collections import OrderedDict
from importlib import import_module

logger = logging.getLogger(__name__)
loggerFront = logging.getLogger('frontend_logs')

def home(request):
    return render(request, 'home.html')

def experiment_config(request):
    methods = approved_methods()
    computers = QuantumComputer.objects.all()
    logger.info("Experiment_config card")
    return render(request, 'experiment_config.html', {'methods': methods, 'computers': computers})

def edit_method(request):
    methods =  all_methods()
    logger.info("Edit_method card")
    return render(request, 'edit_method.html', {'methods': methods})

@csrf_exempt
def start_experiment(request):
    if request.method == "POST" and request.FILES:
        logger.info("Processing POST request")
        processed_results = {}
        images_results = {}
        retrieved_results = {}
        
        try:
            selected_method = request.POST.get('selected_method')
            shots = request.POST.get('shots')
            is_test = request.POST.get('is_test', '').strip().lower() in ['true']
            is_retrieve = request.POST.get('is_retrieve', '').strip().lower() in ['true']

            file_list = request.FILES.getlist('images[]')
            if not file_list:
                file_list = list(request.FILES.values())

            if is_test:
                total = len(file_list)
                passed = 0

            def process_file(uploaded_file):
                uploaded_file.seek(0)
                original_bytes = uploaded_file.read()
                image_base64 = base64.b64encode(original_bytes).decode("utf-8")
                uploaded_file.seek(0)

                unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                file_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
                try:
                    with default_storage.open(file_path, 'wb+') as destination:
                        for chunk in uploaded_file.chunks():
                            destination.write(chunk)
                    logger.info("File saved at: %s", file_path)

                    command = [
                        "geqie",
                        "simulate",
                        "--encoding", selected_method,
                        "--image", file_path,
                        "--grayscale", "false",
                        "--n-shots", shots,
                        "--return-padded-counts", "true"
                    ]
                    logger.info("Executing command: %s", command)
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    logger.info("Command output: %s", result.stdout)
                    output = json.loads(result.stdout)
                    ordered_output = OrderedDict(output)

                    retrieved_image_base64 = None
                    if is_retrieve:
                        encoding_module = import_module(f"geqie.encodings.{selected_method}")
                        retrieved_image = encoding_module.retrieve_function(ordered_output)
                        buf = io.BytesIO()
                        logger.info("Retrieved_image.shape for %s: %s", uploaded_file.name, retrieved_image.shape)
                        if retrieved_image.ndim == 3 and retrieved_image.shape[2] == 3:
                            plt.imsave(buf, retrieved_image, format="png")
                        else:
                            plt.imsave(buf, retrieved_image, cmap="gray", format="png")
                        buf.seek(0)
                        retrieved_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        buf.close()

                    return uploaded_file.name, json.dumps(ordered_output), image_base64, retrieved_image_base64

                except subprocess.CalledProcessError as e:
                    logger.exception("Command failed with return code %s. Stderr: %s", e.returncode, e.stderr)
                    raise Exception(f"Command failed: {e}")
                except json.JSONDecodeError as e:
                    logger.exception("JSON decoding error: %s", e)
                    raise Exception("Invalid JSON returned by the command.")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info("Deleted image at: %s", file_path)

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_file, uploaded_file): uploaded_file
                    for uploaded_file in file_list
                }
                for future in as_completed(futures):
                    file_obj = futures[future]
                    try:
                        file_name, output, orig_b64, retrieved_b64 = future.result()
                        processed_results[file_name] = output
                        images_results[file_name] = orig_b64
                        if retrieved_b64 is not None:
                            retrieved_results[file_name] = retrieved_b64
                        if is_test:
                            passed += 1
                    except Exception as e:
                        logger.exception("Error processing file %s: %s", file_obj.name, e)
                        processed_results[file_obj.name] = "Photo processing error with this method"

            if is_test:
                method_obj = QuantumMethod.objects.get(name=selected_method)
                method_obj.total_tests = total
                method_obj.passed_tests = passed
                method_obj.save()

            final_results = {"processed": processed_results, "image": images_results}
            if is_retrieve:
                final_results["retrieved_image"] = retrieved_results

            logger.info("Returned results: %s", final_results)
            return JsonResponse(final_results)

        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

def read_method_files(request, method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    if not os.path.exists(method_path):
        logger.error("Method not found")
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
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            method_name = data.get("method_name")
            init_content = data.get("init")
            map_content = data.get("map")
            data_content = data.get("data")
            retrieve_content = data.get("retrieve")
            is_new = data.get("is_new", False)
            save_name = data.get("save_name")
            add_new = data.get("add_new")

            if not method_name:
                logger.error("No method name provided")
                return JsonResponse({"error": "No method name provided"}, status=400)

            method_path = os.path.join(ENCODINGS_DIR, method_name)

            if is_new and not os.path.exists(os.path.join(ENCODINGS_DIR, save_name)):
                method_path = os.path.join(ENCODINGS_DIR, save_name)
                os.makedirs(method_path)

            # if add_new:
                files = {
                    "__init__.py": """from .init import init as init_function
from .data import data as data_function
from .map import map as map_function
from .retrieve import retrieve as retrieve_function""",
                    "init.py": init_content,
                    "map.py": map_content,
                    "data.py": data_content,
                    "retrieve.py": retrieve_content,
                }
            # else:
            #     files = {
            #         "init.py": init_content,
            #         "map.py": map_content,
            #         "data.py": data_content,
            #         "retrieve.py": retrieve_content,
            #     }

            for filename, content in files.items():
                file_path = os.path.join(method_path, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                os.chmod(file_path, 0o755)

            refresh_quantum_methods()

            return JsonResponse({"message": "Method saved successfully"})

        except Exception as e:
            logger.exception("Error: %s", str(e))
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

def check_folder_exists(request):
    folder_name = request.GET.get("folder_name", "")
    folder_path = os.path.join(settings.ENCODINGS_DIR, folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return JsonResponse({"exists": True})
    else:
        return JsonResponse({"exists": False})

def get_all_images(request):
    images = []
    try:
        folder_path = os.path.join(settings.MEDIA_ROOT, 'grayscale')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        for file_name in image_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                encoded_data = base64.b64encode(f.read()).decode("utf-8")
            if file_name.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            elif file_name.lower().endswith('.gif'):
                mime_type = "image/gif"
            else:
                mime_type = "image/png"
            images.append({
                "name": file_name,
                "data": encoded_data,
                "type": mime_type,
            })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"images": images})

@csrf_exempt
def log_from_js(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            level = data.get('level', 'info')
            message = data.get('message', '')

            if level == 'debug':
                loggerFront.debug(message)
            elif level == 'info':
                loggerFront.info(message)
            elif level == 'warning':
                loggerFront.warning(message)
            elif level == 'error':
                loggerFront.error(message)
            elif level == 'critical':
                loggerFront.critical(message)
            else:
                loggerFront.info(f"[Unknown level] {message}")

            return JsonResponse({'status': 'success'})
        except Exception as e:
            loggerFront.exception(f"Failed to process JS log: {str(e)}")
            return JsonResponse({'status': 'error'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)


# @csrf_exempt
# def update_list(request):
#     if request.method == "POST":
#         try:
#             print("Running geqie list-encodings...")
#             result = subprocess.run(
#                 ["geqie", "list-encodings"],
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )

#             print(f"Command output: {result.stdout}")
            
#             raw_output = result.stdout.strip()
#             cleaned_output = raw_output.lstrip("[").rstrip("]").replace("'", "").split(", ")

#             for encoding in cleaned_output:
#                 encoding = encoding.strip()
#                 QuantumMethod.objects.get_or_create(name=encoding)

#             print("Encodings successfully added to QuantumMethod")
#             return JsonResponse({"success": True, "message": "Encodings updated successfully!"})
#         except FileNotFoundError:
#             return JsonResponse({"success": False, "error": "geqie.exe not found. Make sure it is in the system PATH."}, status=500)
#         except subprocess.CalledProcessError as e:
#             return JsonResponse({"success": False, "error": f"Command failed: {e}"}, status=500)
#         except Exception as e:
#             return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)
#     else:
#         return JsonResponse({"success": False, "error": "Invalid request method."}, status=405)