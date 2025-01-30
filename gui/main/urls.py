from django.urls import path
from . import views
from .views import start_experiment, update_list
from .views import edit_method_view, read_method_files

urlpatterns = [
    path('', views.experiment_config, name='experiment_config'),
    path('start-Experiment/', start_experiment, name='start_experiment'),
    path('method/edit/', views.edit_method, name='edit_method'),
    path('method/add/', views.add_method, name='add_method'),
    path('update-list/', update_list, name='update_list'),
    path("edit-method/", edit_method_view, name="edit-method"),
    path("get-method/<str:method_name>/", read_method_files, name="get-method"),
]