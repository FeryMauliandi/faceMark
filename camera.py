import cv2
from faceDetection import FaceMesh

faceMesh = FaceMesh(min_detection_confidence=0.7,min_tracking_confidence=0.7)

webcam = cv2.VideoCapture()
webcam.open(0,cv2.CAP_DSHOW)

while True:
    status, frame = webcam.read()
    # frame = cv2.flip(frame,1)
    faceLandmark = faceMesh.findFaceLandMarks(image=frame,draw=True)
    cv2.imshow('face',frame)
    if cv2.waitKey(1) == ord('a'):
        break

cv2.destroyAllWindows
webcam.release()