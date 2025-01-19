from django.contrib import admin
from django.urls import path, include
from .views import home

urlpatterns = [
    path('admin/', admin.site.urls),            # Admin page
    path('face-auth/', include('authentication.urls')),  # Include the app's urls.py
    path('', home),  # Include the app's urls.py
]
