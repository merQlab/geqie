from django.contrib import admin
from .models import QuantumMethod, QuantumSubMethod, QuantumComputer, QuantumSubComputer

class QuantumSubMethodInline(admin.TabularInline):
    model = QuantumSubMethod
    fields = ('name', 'description', 'init', 'map', 'data')

@admin.register(QuantumMethod)
class QuantumMethodAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    inlines = [QuantumSubMethodInline]


class QuantumSubComputerInline(admin.TabularInline):
    model = QuantumSubComputer
    fields = ('name', 'description')

@admin.register(QuantumComputer)
class QuantumComputerAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    inlines = [QuantumSubComputerInline]