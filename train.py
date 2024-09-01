import os
import pickle
import face_recognition

# Define the path to your dataset
DATASET_PATH = 'Dataset\\Input'
ENCODINGS_FILE = 'known_faces.pkl'

def encode_known_faces(dataset_path):
    known_encodings = []
    known_names = []

    # Loop over the directory structure
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        print(person_name)
        i=0
        # Check if it's a directory
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                print(i)
                i+=1
                image_path = os.path.join(person_dir, image_name)
                # Load image and detect faces
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                # Assuming each image has exactly one face
                if len(face_encodings) > 0:
                    encoding = face_encodings[0]
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                else:
                    print(f"No faces found in {image_name}. Skipping...")
    print(known_encodings)
    print(known_names)
    # Save encodings to file
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    
    print(f"Encodings saved to {ENCODINGS_FILE}")

if __name__ == "__main__":
    encode_known_faces(DATASET_PATH)
