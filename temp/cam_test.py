# image test
import cv2

cap = cv2.VideoCapture(0)

while True:
    if cap.grab():
        ret, frame = cap.retrieve(0)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
