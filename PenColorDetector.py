import cv2
import numpy as np


class PenColorDetector:
    def __init__(self):
        # Define the color range for the Hauser pen
        self.low_hauser = np.array([90, 50, 50])  # Lower range of Hauser pen color in HSV
        self.high_hauser = np.array([110, 255, 255])  # Upper range of Hauser pen color in HSV

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask using the color range for the Hauser pen
        mask = cv2.inRange(hsv_img, self.low_hauser, self.high_hauser)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box