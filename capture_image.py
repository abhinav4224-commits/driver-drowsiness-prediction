import cv2
import time

def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()
    return frame
