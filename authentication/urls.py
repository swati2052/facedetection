from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),  # Registration logic
    path('capture/', views.capture, name='capture'),  # Image capture and face authentication
]
