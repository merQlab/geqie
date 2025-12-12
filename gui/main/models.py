import uuid
from django.db import models

class QuantumMethod(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(default='', blank=True)
    init = models.TextField(default='', blank=True)
    map = models.TextField(default='', blank=True)
    data = models.TextField(default='', blank=True)
    retrieve = models.TextField(default='', blank=True)
    total_tests = models.PositiveIntegerField(default=0)
    passed_tests = models.PositiveIntegerField(default=0)
    test = models.CharField(max_length=10, editable=True)
    approved = models.BooleanField("Approved", default=False)

    def __str__(self):
        return self.name

class QuantumComputer(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name
    
class QuantumSubComputer(models.Model):
    quantum_computer = models.ForeignKey(QuantumComputer, on_delete=models.CASCADE, related_name='sub_computers')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    def __str__(self):
        return f"{self.name} ({self.quantum_computer.name})"
    
class Job(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    method = models.CharField(max_length=100)
    shots = models.CharField(max_length=32)
    is_retrieve = models.BooleanField(default=False)

    input_key = models.CharField(max_length=512)
    output_json_key = models.CharField(max_length=512, blank=True, null=True)
    retrieved_png_key = models.CharField(max_length=512, blank=True, null=True)
    original_png_key = models.CharField(max_length=512, blank=True, null=True)

    status = models.CharField(max_length=20, default="queued")
    error = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)