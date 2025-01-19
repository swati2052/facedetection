from django.shortcuts import render
from django.http import HttpResponse

# Home page view (webcam interface)
def home(request):
    return render(request, 'index.html')  # This assumes you have a template 'home.html'

# Register page view
def register(request):
    if request.method == 'POST':
        # Registration logic here (e.g., save user info, etc.)
        return HttpResponse("User registered successfully!")
    return render(request, 'register.html')  # This assumes you have a template 'register.html'

# Capture page view (image capture and face authentication)
def capture(request):
    if request.method == 'POST':
        # Face authentication logic here (e.g., compare faces, etc.)
        return HttpResponse("Face captured and authenticated successfully!")
    return render(request, 'capture.html')  # This assumes you have a template 'capture.html'
