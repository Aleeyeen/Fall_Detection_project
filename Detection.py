import cv2
import numpy as np
from matplotlib import pyplot as plot


cap = cv2.VideoCapture('1.bmp')
frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # range RED color HSV
    lower_blue = np.array([35, 140, 60])
    upper_blue = np.array([255, 255, 180])
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    frame_blur = cv2.GaussianBlur(mask, (5, 5), 1)

    mask_erosion = cv2.erode(frame_blur, None, iterations=2)

    frame_edge = cv2.Canny(mask_erosion, 50, 100)

    cv2.imshow('Mask_erode', mask_erosion)
    cv2.imshow('frame_edge', frame_edge)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()