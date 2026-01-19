import os
import logging

from django.conf import settings
from gui.settings import ENCODINGS_DIR
from .models import QuantumMethod

logger = logging.getLogger(__name__)

def read_method_files(method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    files = {}
    for filename in ["init.py", "map.py", "data.py", "retrieve.py"]:
        key = filename.replace(".py", "")
        file_path = os.path.join(method_path, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    files[key] = f.read()
            else:
                files[key] = ""
        except Exception as e:
            logger.exception("File reading error %s: %s", file_path, e)
            files[key] = ""
    return files

def all_methods():
    methods = []
    if os.path.exists(ENCODINGS_DIR):
        try:
            for method_name in os.listdir(ENCODINGS_DIR):
                method_path = os.path.join(ENCODINGS_DIR, method_name)
                if os.path.isdir(method_path):
                    methods.append({"name": method_name})
        except Exception as e:
            logger.exception("Error listing methods in %s: %s", ENCODINGS_DIR, e)
    else:
        logger.warning("The ENCODINGS_DIR directory does not exist: %s", ENCODINGS_DIR)
    return sorted(methods)

def approved_methods():
    methods = []
    if os.path.exists(ENCODINGS_DIR):
        try:
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
        except Exception as e:
            logger.exception("Error retrieving approved methods: %s", e)
    return sroted(methods)

def refresh_quantum_methods():
    methods = all_methods()
    for method in methods:
        method_name = method["name"]
        file_contents = read_method_files(method_name)
        default_description = f"The implementation of the {method_name} method has been loaded from files."
        try:
            qs = QuantumMethod.objects.filter(name=method_name)
            if qs.count() > 1:
                first = qs.first()
                qs.exclude(pk=first.pk).delete()

            obj, created = QuantumMethod.objects.get_or_create(name=method_name)
            if created:
                obj.description = default_description
            obj.init = file_contents.get("init", "")
            obj.map = file_contents.get("map", "")
            obj.data = file_contents.get("data", "")
            obj.retrieve = file_contents.get("retrieve", "")
            update_fields = ["init", "map", "data", "retrieve"]
            if created:
                update_fields.insert(0, "description")
            obj.save(update_fields=update_fields)
        except Exception as e:
            logger.exception("Method refresh error '%s': %s", method_name, e)

def update_method_files(instance, old_name=None):
    new_path = os.path.join(ENCODINGS_DIR, instance.name)
    try:
        if old_name and old_name != instance.name:
            old_path = os.path.join(ENCODINGS_DIR, old_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            else:
                os.makedirs(new_path, exist_ok=True)
        else:
            if not os.path.exists(new_path):
                os.makedirs(new_path, exist_ok=True)
    except Exception as e:
        logger.exception("Error creating/changing method directory: %s", e)
        return

    try:
        files = settings.DEFAULT_INIT.copy() if hasattr(settings, "DEFAULT_INIT") else {}
        files.update({
            "init.py": instance.init or "",
            "map.py": instance.map or "",
            "data.py": instance.data or "",
            "retrieve.py": instance.retrieve or "",
        })

        for filename, content in files.items():
            file_path = os.path.join(new_path, filename)
            try:
                with open(file_path, "w", encoding="utf-8", newline='') as f:
                    f.write(content)
                os.chmod(file_path, 0o755)
            except Exception as e:
                logger.exception("File saving error %s: %s", file_path, e)

    except Exception as e:
        logger.exception("Error updating method files in directory %s: %s", new_path, e)

    refresh_quantum_methods()