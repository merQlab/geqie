from django.shortcuts import render
from django.http import JsonResponse

from gui.settings import ENCODINGS_DIR
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from collections import OrderedDict
import json
import subprocess
import os
import uuid
import logging

logger = logging.getLogger(__name__)
loggerFront = logging.getLogger('frontend_logs')

def home(request):
    return render(request, 'home.html')

def experiment_config(request):
    methods = list_methods()
    computers = QuantumComputer.objects.all()
    logger.info("Experiment_config card")
    return render(request, 'experiment_config.html', {'methods': methods, 'computers': computers})

@csrf_exempt
def start_experiment(request):
    if request.method == "POST" and request.FILES:
        logger.info("Processing POST request")
        results = {}
        try:
            selected_method = request.POST.get('selected_method')
            shots = request.POST.get('shots')

            for key, uploaded_file in request.FILES.items():

                unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                file_path = os.path.join(settings.MEDIA_ROOT, unique_filename)

                with default_storage.open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                logger.info("File saved at: %s", str(file_path))

                command = [
                    "geqie",
                    "simulate",
                    "--encoding", selected_method,
                    "--image", file_path,
                    "--n-shots", shots,
                    "--return-padded-counts", "true"
                ]
                logger.info("Executing command: %s", str(command))

                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    logger.info("Command output: %s", str(result.stdout))

                    output = json.loads(result.stdout)
                    results[uploaded_file.name] = output
                    os.remove(file_path)
                    logger.info("Deleted image")

                except subprocess.CalledProcessError as e:
                    logger.critical("Command failed with return code %s. Stderr: %s", str(e.returncode), str(e.stderr))
                    os.remove(file_path)
                    return JsonResponse({"success": False, "error": f"Command failed: {e}"}, status=500)

                except json.JSONDecodeError as e:
                    logger.critical("JSON decoding error: %s", str(e))
                    os.remove(file_path)
                    return JsonResponse({"success": False, "error": "Invalid JSON returned by the command."}, status=500)

            logger.info("Returned results: %s", str(results))
            return JsonResponse(results)

        except Exception as e:
            logger.critical("Unexpected error: %s", str(e))
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

def edit_method(request):
    methods =  list_methods()
    logger.info("Edit_method card")
    return render(request, 'edit_method.html', {'methods': methods})

def list_methods():
    methods = []
    if os.path.exists(ENCODINGS_DIR):
        for method_name in os.listdir(ENCODINGS_DIR):
            method_path = os.path.join(ENCODINGS_DIR, method_name)
            if os.path.isdir(method_path):
                methods.append({"name": method_name})
    return methods

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