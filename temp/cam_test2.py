from imutils.video import VideoStream
import cv2

cap_api = cv2.CAP_ANY
webcam = cv2.VideoCapture("/dev/video3", cv2.CAP_V4L2)
webcam.set(3, 640)
webcam.set(4,480)
cv2.waitKey(2)
cam2 = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
cam2.set(3, 640)
cam2.set(4,480)
cv2.waitKey(2)

while True:
	if webcam.grab():
		sucess, img = webcam.retrieve(0)
		if img is not None:
			cv2.imshow("Webcam", img)
	if cam2.grab():
		sucess2, img2 = cam2.retrieve(0)
		if img2 is not None:
			cv2.imshow("Webcam 2", img2)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
