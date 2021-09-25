import cv2
import PoseModule as pm
import time
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

kernel = None
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    # Read a new frame
    ret, img = cap.read()
    # Check if frame is not read correctly
    if not ret:
        break
    # apply the background object on each frame to get the segment mask
    fgmask = backgroundobject.apply(img)
    # Perform thresholding to get rid of the shadows.
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Apply some morphological operations to mask sure you have a good mask
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    # also the extracting the real detected foreground part of the image (optional)
    # Detect contours in the frame
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a copy of the frame to draw bounding box around the detected
    frameCopy = img.copy()
    # loop over each contour found in the frame
    for cnt in contours:
        if cv2.contourArea(cnt) > 350:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frameCopy, "Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    foregroundPart = cv2.bitwise_and(img, img, mask=fgmask)
    stacked = np.hstack((img, foregroundPart, frameCopy))
    cv2.imshow("Original Frame, Extracted Foreground and Detected", cv2.resize(stacked, None, fx=0.65, fy=0.65))

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[0])
        cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1.0 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow('image', img)

    key = cv2.waitKey(40)
    if key == 27:
        break

cap.release()






