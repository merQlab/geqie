from django.urls import path
from . import views
from .views import start_experiment, job_status, read_method_files, save_method_files, check_folder_exists, get_all_images, log_from_js

urlpatterns = [
    path("", views.experiment_config, name="experiment_config"),

    path("job-status/<uuid:job_id>/", job_status, name="job_status"),
    path("job-status/<uuid:job_id>", job_status),
    path("jobs/<uuid:job_id>/", job_status),
    path("jobs/<uuid:job_id>", job_status),
    path("start-experiment/", start_experiment, name="start_experiment"),

    path("get-all-images/", get_all_images, name="get_all_images"),
    path("media/proxy/<str:token>/", views.proxy_file, name="proxy-file"),

    path("get-method/<str:method_name>/", read_method_files, name="get-method"),
    path("methods/", views.edit_method, name="edit_method"),
    path("save-method/", save_method_files, name="save-method"),

    path("logs/", log_from_js, name="log_from_js"),
    path("check-folder/", check_folder_exists, name="check_folder"),
]