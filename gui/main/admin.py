from django.contrib import admin
from .utils import refresh_quantum_methods
from .models import QuantumMethod, QuantumComputer, QuantumSubComputer
import os, logging, shutil
from gui.settings import ENCODINGS_DIR

logger = logging.getLogger(__name__)

@admin.register(QuantumMethod)
class QuantumMethodAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'test', 'approved')
    fields = ('name', 'description', 'init', 'map', 'data', 'approved')

    def get_queryset(self, request):
        refresh_quantum_methods()
        return super().get_queryset(request)

    def test(self, obj):
        try:
            if obj.total_tests and obj.total_tests > 0:
                percent = (obj.passed_tests / obj.total_tests) * 100
                if percent.is_integer():
                    return f"{int(percent)}%"
                else:
                    return f"{percent:.2f}%"
        except AttributeError:
            pass
        return "0%"
    test.short_description = "Test"

    def delete_queryset(self, request, queryset):
        for obj in queryset:
            method_folder = os.path.join(ENCODINGS_DIR, obj.name)
            if os.path.exists(method_folder):
                shutil.rmtree(method_folder)
        queryset.delete()

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