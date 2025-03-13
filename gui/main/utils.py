import os
from gui.settings import ENCODINGS_DIR
from .models import QuantumMethod

def refresh_quantum_methods():
    methods = all_methods()
    for method in methods:
        method_name = method["name"]
        file_contents = read_method_files(method_name)
        description = f"The implementation of the {method_name} method has been loaded from files."

        qs = QuantumMethod.objects.filter(name=method_name)
        if qs.count() > 1:
            first = qs.first()
            qs.exclude(pk=first.pk).delete()

        QuantumMethod.objects.update_or_create(
            name=method_name,
            defaults={
                "description": description,
                "init": file_contents.get("init", ""),
                "map": file_contents.get("map", ""),
                "data": file_contents.get("data", ""),
                "retrieve": file_contents.get("retrieve", "")
            }
        )

def read_method_files(method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    files = {}
    for filename in ["init.py", "map.py", "data.py", "retrieve.py"]:
        key = filename.replace(".py", "")
        file_path = os.path.join(method_path, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                files[key] = f.read()
        else:
            files[key] = ""
    return files

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
                qm = QuantumMethod.objects.filter(name=method_name, approved=True).first()
                if qm:
                    methods.append({
                        "id": qm.id,
                        "name": qm.name,
                        "description": qm.description
                    })
    return methods