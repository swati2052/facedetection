import cv2

video_capture =cv2.VideoCapture(0)

while True:
    _,  img = video_capture.read()
    cv2.imshow("face detection",img)
    if (cv2.waitKey(30) & 0xFF) == 27:
        break
    # cv2.imshow("face detection")   
video_capture.release()