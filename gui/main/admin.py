from django.contrib import admin
from .models import QuantumMethod, QuantumComputer, QuantumSubComputer
import os, logging, shutil
from gui.settings import ENCODINGS_DIR
from .views import all_methods

logger = logging.getLogger(__name__)

def read_method_files(method_name):
    method_path = os.path.join(ENCODINGS_DIR, method_name)
    files = {}
    for filename in ["init.py", "map.py", "data.py"]:
        key = filename.replace(".py", "")
        file_path = os.path.join(method_path, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                files[key] = f.read()
        else:
            files[key] = ""
    return files

@admin.register(QuantumMethod)
class QuantumMethodAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'approved')
    fields = ('name', 'description', 'init', 'map', 'data', 'approved')

    def get_queryset(self, request):
        self.refresh_methods()
        return super().get_queryset(request)

    def refresh_methods(self):
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
                    "data": file_contents.get("data", "")
                }
            )

    def delete_model(self, request, obj):
        method_folder = os.path.join(ENCODINGS_DIR, obj.name)
        if os.path.exists(method_folder):
            shutil.rmtree(method_folder)
        super().delete_model(request, obj)

class QuantumSubComputerInline(admin.TabularInline):
    model = QuantumSubComputer
    fields = ('name', 'description')

@admin.register(QuantumComputer)
class QuantumComputerAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    inlines = [QuantumSubComputerInline]