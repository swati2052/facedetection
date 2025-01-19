from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import base64
import numpy as np
import face_recognition
from .models import User

import cv2  # OpenCV for webcam and image processing

# View for the home page with webcam interface
def home(request):
    return render(request, 'index.html')  # This would render your webcam interface page

# View for the registration page logic
def register(request):
    if request.method == 'POST':
        # Registration logic (e.g., save face image, user details)
        # Example: Save the captured face image for registration

        name = request.POST.get('name')
        email = request.POST.get('email')
        passkey = request.POST.get('password')
        base64_string = request.POST.get('image')

        # Extract the image type and base64 data
        header, encoded = base64_string.split(',', 1)

        # Decode the base64 string
        image_data = base64.b64decode(encoded)

        try:
            user_data = User(
                    username= name,
                    email=email,
                    password_hash=passkey,
                    face_data=image_data
                )
            user_data.save()
        except Exception:
            return HttpResponseBadRequest("Error")
        return HttpResponse('Registration successful.')
    return render(request, 'index.html')  # Render the registration form

# View for capturing an image and performing face authentication
def capture(request):
    if request.method == 'POST':
        # Logic for capturing the face image
        # Use OpenCV or any method to access the webcam and capture the image
        # This is where you might integrate face recognition
        name = request.POST.get('name')
        is_passAuth = request.POST.get('method')
        user = User.objects.filter(username=name)
        if not user:
            return HttpResponse('No such user')

        user = user[0]
        if is_passAuth:
            passkey = request.POST.get('password')
            if user.password_hash == passkey:
                return HttpResponse('Login')
            else:
                return HttpResponse('Invalid credentials')
        else:
            base64_string = request.POST.get('image')
            header, encoded = base64_string.split(',', 1)

            # Decode the base64 string
            image_data = base64.b64decode(encoded)
            user_image_data = user.face_data
            # print(image_data)
            nparr = np.frombuffer(image_data, np.uint8)
            user_nparr = np.frombuffer(user_image_data, np.uint8)

            # Decode the NumPy array to an image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            user_image = cv2.imdecode(user_nparr, cv2.IMREAD_COLOR)

            # Convert BGR (OpenCV format) to RGB (face_recognition expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            user_image_rgb = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

            # Process the image with face_recognition
            face_locations = face_recognition.face_locations(image_rgb)
            user_face_locations = face_recognition.face_locations(user_image_rgb)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            user_face_data = face_recognition.face_encodings(user_image, user_face_locations)

            # Assuming each image has exactly one face
            # if len(face_encodings) > 0:
            encoding = face_encodings[0]
        
            matches = face_recognition.compare_faces(user_face_data, encoding)
            face_distances = face_recognition.face_distance(user_face_data, encoding)
            if matches[0]:
                return HttpResponse('Login')
            else:
                return HttpResponse('Invalid credentials')           
            
            
    return render(request, 'index.html')  # Render the capture page where the webcam is shown
