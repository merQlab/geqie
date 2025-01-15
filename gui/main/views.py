from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import QuantumMethod, QuantumComputer
from django.views.decorators.csrf import csrf_exempt
import json

def home(request):
    return render(request, 'home.html')

def experiment_config(request):
    methods = QuantumMethod.objects.all()
    computers = QuantumComputer.objects.all()
    
    return render(request, 'experiment_config.html', {'methods': methods, 'computers': computers})

@csrf_exempt
def start_experiment(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            folder_path = data.get('folder')

            if not folder_path:
                return JsonResponse({"error": "No folder path provided"}, status=400)

            print("Received folder path:", folder_path)

            return JsonResponse({"message": "Folder path received successfully!"}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)

def edit_method(request):
    methods = QuantumMethod.objects.all()
    return render(request, 'edit_method.html', {'methods': methods})

def add_method(request):
    methods = QuantumMethod.objects.all()
    return render(request, 'add_method.html', {'methods': methods})