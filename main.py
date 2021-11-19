import cv2
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture(0)
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)


h = [0, 0]
w = [0, 0]
frames = []
detector = pm.poseDetector()
y = [0, 0]

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

frames_zr = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = detector.findPose(orig_frame)
        lmList = detector.findPosition(img, draw=False)


        diff_frame = cv2.absdiff(gray, grayMedianFrame)
        _, fgmask = cv2.threshold(diff_frame, 85, 255, cv2.THRESH_BINARY)

        kernel = np.ones((10, 10), np.uint8)
        fgmask = cv2.dilate(fgmask, kernel, iterations=4)
        fgmask = cv2.erode(fgmask, kernel, iterations=3)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=4)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Area = 0

        for c in contours:
            vertices = cv2.boundingRect(c)
            if(cv2.contourArea(c) > Area):
                Area = cv2.contourArea(c)
                rectangle = vertices

                point1 = (rectangle[0], rectangle[1])
                point2 = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
                cv2.rectangle(orig_frame, point1, point2, (0, 255, 0), 1)
                cv2.rectangle(fgmask, point1, point2, (255, 255, 255), 1)

                w.append(rectangle[2])
                h.append(rectangle[3])

                if len(lmList) != 0:
                    cy = lmList[0]
                    y.append(cy)

                    if((w[-2]/rectangle[2] > 1) and (h[-2]/rectangle[3] > 1) and ((cy - y[-2])) > 60):
                        cv2.putText(orig_frame, "Fall detect", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, 11)
                        print("fall detection")

            cv2.imshow('frame_detect', orig_frame)
            cv2.imshow('frame', fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()