# Create your models here.
from django.db import models
from django.utils.timezone import now

class User(models.Model):
    username = models.CharField(max_length=50, unique=True)  # Unique username
    email = models.EmailField(unique=True)  # Unique email
    password_hash = models.CharField(max_length=255)  # Password hash for security
    face_data = models.BinaryField()  # Binary data for storing face encoding or image
    created_at = models.DateTimeField(default=now)  # Automatically records creation time
    updated_at = models.DateTimeField(auto_now=True)  # Automatically updates on record modification

    def __str__(self):
        return self.username


class LoginAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to the User table
    attempt_time = models.DateTimeField(default=now)  # Time of login attempt
    success = models.BooleanField(default=False)  # Whether the login was successful

    def __str__(self):
        return f"Login Attempt by {self.user.username} - {'Success' if self.success else 'Failed'}"
