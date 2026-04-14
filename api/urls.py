from django.urls import path
from .views import unir_imagenes_view

urlpatterns = [
    path('stitch/', unir_imagenes_view),
]