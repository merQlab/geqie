from django.shortcuts import render
from django.http import JsonResponse
from gui.settings import ENCODINGS_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import subprocess
import os
import uuid
import logging
from collections import OrderedDict

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
    methods =  approved_methods()
    logger.info("Edit_method card")
    return render(request, 'edit_method.html', {'methods': methods})

@csrf_exempt
def start_experiment(request):
    if request.method == "POST" and request.FILES:
        logger.info("Processing POST request")
        results = {}
        try:
            selected_method = request.POST.get('selected_method')
            shots = request.POST.get('shots')

            def process_file(uploaded_file):
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
                        "--n-shots", shots,
                        "--return-padded-counts", "true"
                    ]
                    logger.info("Executing command: %s", command)
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    logger.info("Command output: %s", result.stdout)

                    output = json.loads(result.stdout)
                    logger.info("Command output: %s", output)
                    ordered_output = json.dumps(OrderedDict(output))
                    logger.info("OrderedDict: %s", ordered_output)
                    
                    return uploaded_file.name, ordered_output
                except subprocess.CalledProcessError as e:
                    logger.critical("Command failed with return code %s. Stderr: %s", e.returncode, e.stderr)
                    raise Exception(f"Command failed: {e}")
                except json.JSONDecodeError as e:
                    logger.critical("JSON decoding error: %s", e)
                    raise Exception("Invalid JSON returned by the command.")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info("Deleted image at: %s", file_path)

            file_list = request.FILES.getlist('images[]')
            if not file_list:
                file_list = list(request.FILES.values())

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_file, uploaded_file): uploaded_file
                    for uploaded_file in file_list
                }
                for future in as_completed(futures):
                    try:
                        file_name, output = future.result()
                        results[file_name] = output
                    except Exception as e:
                        logger.critical("Error processing file: %s", e)
                        return JsonResponse({"success": False, "error": str(e)}, status=500)

            logger.info("Returned results: %s", results)
            return JsonResponse(results)

        except Exception as e:
            logger.critical("Unexpected error: %s", e)
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

def all_methods():
    methods = []
    if os.path.exists(ENCODINGS_DIR):
        for method_name in os.listdir(ENCODINGS_DIR):
            method_path = os.path.join(ENCODINGS_DIR, method_name)
            if os.path.isdir(method_path):
                methods.append({"name": method_name})
    return methods

def approved_methods():
    methods = []
    if os.path.exists(ENCODINGS_DIR):
        for method_name in os.listdir(ENCODINGS_DIR):
            method_path = os.path.join(ENCODINGS_DIR, method_name)
            if os.path.isdir(method_path):
                methods.append({"name": method_name})

        approved_method_names = list(
        QuantumMethod.objects.filter(approved=True).values_list('name', flat=True)
        )
        approved_methods = [m for m in methods if m["name"] in approved_method_names]
    return approved_methods

def read_method_files(request, method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    if not os.path.exists(method_path):
        logger.error("Method not found")
        return JsonResponse({"error": "Method not found"}, status=404)

    files = ["init.py", "map.py", "data.py"]
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

            if add_new:
                files = {
                    "__init__.py": """from .init import init as init_function
from .data import data as data_function
from .map import map as map_function""",
                    "init.py": init_content,
                    "map.py": map_content,
                    "data.py": data_content,
                }
            else:
                files = {
                    "init.py": init_content,
                    "map.py": map_content,
                    "data.py": data_content,
                }

            for filename, content in files.items():
                with open(os.path.join(method_path, filename), "w", encoding="utf-8") as f:
                    f.write(content)

            return JsonResponse({"message": "Method saved successfully"})

        except Exception as e:
            logger.critical("Error: %s", str(e))
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

def check_folder_exists(request):
    folder_name = request.GET.get("folder_name", "")
    folder_path = os.path.join(settings.ENCODINGS_DIR, folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return JsonResponse({"exists": True})
    else:
        return JsonResponse({"exists": False})

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
            loggerFront.error(f"Failed to process JS log: {str(e)}")
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