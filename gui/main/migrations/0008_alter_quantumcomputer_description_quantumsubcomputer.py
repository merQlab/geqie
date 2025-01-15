# Generated by Django 5.1.4 on 2025-01-13 13:21

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0007_alter_quantummethod_description'),
    ]

    operations = [
        migrations.AlterField(
            model_name='quantumcomputer',
            name='description',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.CreateModel(
            name='QuantumSubComputer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True, default='')),
                ('quantum_computer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='sub_computers', to='main.quantumcomputer')),
            ],
        ),
    ]
