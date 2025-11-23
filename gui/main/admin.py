import os
import logging
import shutil

from django.contrib import admin
from gui.settings import ENCODINGS_DIR
from .models import QuantumMethod, QuantumComputer, QuantumSubComputer
from .utils import refresh_quantum_methods, update_method_files

logger = logging.getLogger(__name__)

@admin.register(QuantumMethod)
class QuantumMethodAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'test', 'approved')
    fields = ('name', 'description', 'init', 'map', 'data', 'retrieve', 'test', 'approved')

    def get_queryset(self, request):
        refresh_quantum_methods()
        return super().get_queryset(request)

    def delete_queryset(self, request, queryset):
        for obj in queryset:
            method_folder = os.path.join(ENCODINGS_DIR, obj.name)
            if os.path.exists(method_folder):
                try:
                    shutil.rmtree(method_folder)
                    logger.info("Method folder deleted: %s", method_folder)
                except Exception as e:
                    logger.exception("Error deleting folder %s: %s", method_folder, e)
        queryset.delete()

    def delete_model(self, request, obj):
        method_folder = os.path.join(ENCODINGS_DIR, obj.name)
        if os.path.exists(method_folder):
            try:
                shutil.rmtree(method_folder)
                logger.info("Method folder deleted: %s", method_folder)
            except Exception as e:
                logger.exception("Error deleting folder %s: %s", method_folder, e)
        super().delete_model(request, obj)
    
    def compute_test_value(self, obj):
        try:
            if obj.total_tests and obj.total_tests > 0:
                percent = (obj.passed_tests / obj.total_tests) * 100
                return f"{int(percent)}%" if percent.is_integer() else f"{percent:.2f}%"
        except (AttributeError, ZeroDivisionError):
            logger.exception("Error calculating test result for method %s", obj.name)
        return "0%"

    def changelist_view(self, request, extra_context=None):
        qs = self.get_queryset(request)
        for obj in qs:
            computed = self.compute_test_value(obj)
            if not obj.test or obj.test == computed:
                obj.test = computed
                obj.save(update_fields=['test'])
        return super().changelist_view(request, extra_context=extra_context)
    
    def save_model(self, request, obj, form, change):
        old_name = None
        if change:
            try:
                old_instance = QuantumMethod.objects.get(pk=obj.pk)
                old_name = old_instance.name
            except QuantumMethod.DoesNotExist:
                logger.warning("The old method record was not found during the update.")
        super().save_model(request, obj, form, change)
        update_method_files(obj, old_name)

class QuantumSubComputerInline(admin.TabularInline):
    model = QuantumSubComputer
    fields = ('name', 'description')

@admin.register(QuantumComputer)
class QuantumComputerAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    inlines = [QuantumSubComputerInline]