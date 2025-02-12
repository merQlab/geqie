from django.db import models

class QuantumMethod(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(default='', blank=True)
    init = models.TextField(default='', blank=True)
    map = models.TextField(default='', blank=True)
    data = models.TextField(default='', blank=True)
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