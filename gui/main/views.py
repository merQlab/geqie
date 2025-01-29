from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import subprocess
import os
import shutil
import time
import uuid

def home(request):
    return render(request, 'home.html')

def experiment_config(request):
    methods = QuantumMethod.objects.all()
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
                file_path = os.path.join('assets', 'test_images', 'grayscale', unique_filename)

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

        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            return JsonResponse({"success": False, "error": "geqie not found. Make sure it is in the system PATH."}, status=500)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)

    return JsonResponse({"success": False, "error": "Invalid request."}, status=400)

@csrf_exempt
def update_list(request):
    if request.method == "POST":
        try:
            print("Running geqie list-encodings...")
            result = subprocess.run(
                ["geqie", "list-encodings"],
                capture_output=True,
                text=True,
                check=True
            )

            print(f"Command output: {result.stdout}")
            
            raw_output = result.stdout.strip()
            cleaned_output = raw_output.lstrip("[").rstrip("]").replace("'", "").split(", ")

            for encoding in cleaned_output:
                encoding = encoding.strip()
                QuantumMethod.objects.get_or_create(name=encoding)

            print("Encodings successfully added to QuantumMethod")
            return JsonResponse({"success": True, "message": "Encodings updated successfully!"})
        except FileNotFoundError:
            return JsonResponse({"success": False, "error": "geqie.exe not found. Make sure it is in the system PATH."}, status=500)
        except subprocess.CalledProcessError as e:
            return JsonResponse({"success": False, "error": f"Command failed: {e}"}, status=500)
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Unexpected error: {e}"}, status=500)
    else:
        return JsonResponse({"success": False, "error": "Invalid request method."}, status=405)

def edit_method(request):
    methods = QuantumMethod.objects.all()
    return render(request, 'edit_method.html', {'methods': methods})

def add_method(request):
    methods = QuantumMethod.objects.all()
    return render(request, 'add_method.html', {'methods': methods})

