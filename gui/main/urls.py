from django.urls import path
from . import views
from .views import start_experiment, read_method_files, save_method_files, check_folder_exists, log_from_js

urlpatterns = [
    path('', views.experiment_config, name='experiment_config'),
    path('start-Experiment/', start_experiment, name='start_experiment'),
    path('method/edit/', views.edit_method, name='edit_method'),
    path("get-method/<str:method_name>/", read_method_files, name="get-method"),
    path("save-method/", save_method_files, name="save-method"),
    path("check_folder/", check_folder_exists, name="check_folder"),
    path('api/log/', log_from_js, name='log_from_js'),
]