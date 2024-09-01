import face_recognition
import pickle
import cv2

# Load known faces and encodings
with open('known_faces.pkl', 'rb') as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

def recognize_face(image_path):
    # Load the input image and convert from BGR to RGB
    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the input image and get encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Initialize a list to store the names of recognized faces
    names = []

    # Loop over the detected faces
    for encoding in face_encodings:
        # Compare face encodings against known faces
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        # Get the best match index
        best_match_index = face_distances.argmin() if matches else None

        # Determine if there's a match
        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw the label with the name below the face
        cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show the image with the recognized faces
    cv2.imshow("Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_path = 'suraj.jpg'  # Provide the local image path here
    recognize_face(test_image_path)
    # test_image_path = 'ali.jpg'  # Provide the local image path here
    # recognize_face(test_image_path)
    # test_image_path = 'alia.jpeg'  # Provide the local image path here
    # recognize_face(test_image_path)
