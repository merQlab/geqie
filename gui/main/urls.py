from django.urls import path
from . import views
from .views import start_experiment

urlpatterns = [
    path('', views.experiment_config, name='experiment_config'),
    path('start-Experiment/', start_experiment, name='start_experiment'),
    path('method/edit/', views.edit_method, name='edit_method'),
    path('method/add/', views.add_method, name='add_method'),
]