import face_recognition
import pickle
import cv2

# Load known faces and encodings
with open('known_faces.pkl', 'rb') as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

def recognize_faces_from_webcam():
    # Initialize the webcam feed
    video_capture = cv2.VideoCapture(0)  # 0 is the default camera

    while True:
        # Capture a single frame of video
        ret, frame = video_capture.read()

        # Resize frame to speed up processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

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
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw the label with the name below the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Face Recognition', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_from_webcam()
