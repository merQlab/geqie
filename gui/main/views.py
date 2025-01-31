from django.shortcuts import render
from django.http import JsonResponse

from gui.settings import ENCODINGS_DIR
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import subprocess
import os
import uuid

def home(request):
    return render(request, 'home.html')

def experiment_config(request):
    methods = list_methods()
    computers = QuantumComputer.objects.all()
    
    return render(request, 'experiment_config.html', {'methods': methods, 'computers': computers})

@csrf_exempt
def start_experiment(request):
    if request.method == "POST" and request.FILES:
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

                print(f"File saved at: {file_path}")

                command = [
                    "geqie",
                    "simulate",
                    "--encoding", selected_method,
                    "--image", file_path,
                    "--n-shots", shots,
                    "--return-padded-counts", "true"
                ]
                print(f"Executing command: {' '.join(command)}")

                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    print(f"Command output: {result.stdout}")

                    output = json.loads(result.stdout.strip())
                    results[uploaded_file.name] = output
                    os.remove(file_path)

                except subprocess.CalledProcessError as e:
                    print(f"Command failed with return code {e.returncode}. Stderr: {e.stderr}") 
                    return JsonResponse({"success": False, "error": f"Command failed: {e}"}, status=500)

                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}") 
                    return JsonResponse({"success": False, "error": "Invalid JSON returned by the command."}, status=500)

            print("Returning results.")
            return JsonResponse(results, safe=False)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

def edit_method(request):
    methods =  list_methods()
    return render(request, 'edit_method.html', {'methods': methods})

def add_method(request):
    methods = QuantumMethod.objects.all()
    return render(request, 'add_method.html', {'methods': methods})

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

            if not method_name:
                return JsonResponse({"error": "No method name provided"}, status=400)

            method_path = os.path.join(ENCODINGS_DIR, method_name)

            if is_new and not os.path.exists(os.path.join(ENCODINGS_DIR, save_name)):
                method_path = os.path.join(ENCODINGS_DIR, save_name)
                os.makedirs(method_path)

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
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

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