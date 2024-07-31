# It helps in identifying the faces
import cv2, sys, numpy, os

size = 4
haar_file = "haarcascade_frontalface_default.xml"
datasets = r"Dataset\Input"

# Part 1: Create fisherRecognizer
print("Recognizing Face Please Be in sufficient Lights...")

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
target_size = (200, 300)
for subdirs, dirs, files in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + "/" + filename
			label = id
			img = cv2.imread(path, 0)
			# Resize the image to the target size
			if img is not None:
				resized_img = cv2.resize(img, target_size)
				images.append(resized_img)
				labels.append(int(label))
		id += 1
		
(width, height) = (130, 100)
# print(names)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
# images_np = numpy.array(images)
# print(set(map(lambda x: x.shape, images)))
# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)

test_image_path = "ali.jpg"

# Load the test image in grayscale
test_image = cv2.imread(test_image_path, 0)

# Ensure the test image was loaded correctly
if test_image is None:
	print("Error: Test image not found or could not be loaded.")
else:
	# Resize the image to the target size
	# target_size = (200, 200)  # Replace with your training image size
	resized_test_image = cv2.resize(test_image, target_size)

	cv2.imshow('img', resized_test_image)
	cv2.waitKey()
	# Predict the label of the test image
	label, confidence = model.predict(resized_test_image)

	# Map the label to the corresponding name
	predicted_name = names.get(label, "Unknown")

	# Output the prediction and confidence
	print(f"Predicted label: {label}")
	print(f"Confidence score: {confidence}")
	print(f"Predicted name: {predicted_name}")

'''




webcam = cv2.VideoCapture(0)
while True:
	(_, im) = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for x, y, w, h in faces:
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
		face = gray[y : y + h, x : x + w]
		face_resize = cv2.resize(face, (width, height))
		# Try to recognize the face
		prediction = model.predict(face_resize)
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

		if prediction[1] < 500:
			cv2.putText(
				im,
				"% s - %.0f" % (names[prediction[0]], prediction[1]),
				(x - 10, y - 10),
				cv2.FONT_HERSHEY_PLAIN,
				1,
				(0, 255, 0),
			)
		else:
			cv2.putText(
				im,
				"not recognized",
				(x - 10, y - 10),
				cv2.FONT_HERSHEY_PLAIN,
				1,
				(0, 255, 0),
			)

	cv2.imshow("OpenCV", im)

	key = cv2.waitKey(10)
	if key == 27:
		break
'''