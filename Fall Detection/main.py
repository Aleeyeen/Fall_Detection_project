import cv2
import numpy as np
import PoseModule as pm
import requests
import time
import datetime

url = 'https://notify-api.line.me/api/notify'
    #token Personal
token = 'iLm162K7YhYixQYoEKcinQbqxHOCcYKEyNEahtEkDci'
    #token Family
#token = 'iszDQGSWKf62Ot51CGDXdD2iLFxuiq8D2kuBjbye0b6'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer '+token}
h = []
w = []
y = []

cap = cv2.VideoCapture(0)
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frames = []
detector = pm.poseDetector()

frames_zr = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

frames_zr = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:

        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = detector.findPose(orig_frame)
        lmList = detector.findPosition(img, draw=False)
        #cv2.imshow('gray', gray)

        diff_frame = cv2.absdiff(gray, grayMedianFrame)
        _, fgmask = cv2.threshold(diff_frame, 80, 255, cv2.THRESH_BINARY)
        # fgmask = cv2.adaptiveThreshold(diff_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 195, 5)
        kernel = np.ones((10, 10), np.uint8)
        # 
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        fgmask = cv2.erode(fgmask, kernel, iterations=3)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=5)
        # fgmask = cv2.erode(fgmask, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Area = 0

        for c in contours:
            vertices = cv2.boundingRect(c)
            if(cv2.contourArea(c) > Area):
                Area = cv2.contourArea(c)
                rectangle = vertices

                point1 = (rectangle[0], rectangle[1])
                point2 = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
                cv2.rectangle(orig_frame, point1, point2, (0, 255, 0), 2)
                cv2.rectangle(fgmask, point1, point2, (255, 255, 255), 1)

                wide = ((rectangle[0] + rectangle[2]) - rectangle[0])
                high = ((rectangle[1] + rectangle[3]) - rectangle[1])

                # w.append(wide)
                # h.append(high)
                # print('w', w)
                # print('rec0 ',rectangle[0])
                # print('rec1 ',rectangle[1])
                # print('rec2 ',rectangle[2])
                # print('rec3 ',rectangle[3])
                # print('point1',point1)
                # print('point2 ',point2)
                # print('hi' ,hi)
                # print('wide ',wide)
                # print('high ',high)
                # r = high/wide
                # print('r ', r)

                if len(lmList) != 0:
                    cy = lmList[0]
                    y.append(cy)
                    # print('y',y)
                    # r = (cy - y[len(y)-5])
                    # print('r',r)

                    #กรณีคิดความกว้างของกล่อง bounding box ทำนายการเกิดการหกล้ม
                    # if (((w[len(w)-2] / rectangle[2] > 1) and h[-2] / rectangle[3] > 1) and (cy - y[-2]) > 70):
                    #     if (prediction[0][0] < prediction[0][1]):
                    #         cv2.putText(orig_frame, "Fall detect", (rectangle[0], rectangle[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #         cv2.rectangle(orig_frame, point1, point2, (0, 0, 255), 2)
                    #         print("fall detection")

                    if(high < wide):
                        # cv2.rectangle(orig_frame, point1, point2, (0, 0, 255), 2)
                        try:
                            r = cy - y[len(y)-3]
                            if(r > 44.10):
                                cv2.putText(orig_frame, "Fall detect", (rectangle[0], rectangle[1] - 7), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.rectangle(orig_frame, point1, point2, (0, 0, 255), 2)
                                print("fall detection")
                                r = requests.post(url, headers=headers, data={'message': 'Test Fall Detect System'})
                                print(r.text)
                               
                        except:
                            cv2.putText(orig_frame, 'Not Fall', (rectangle[0], rectangle[1] - 7), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(orig_frame, point1, point2, (0, 255, 0), 2)
                        # else:
                        #     cv2.rectangle(orig_frame, point1, point2, (0, 255, 0), 2)
                        #     cv2.putText(orig_frame, 'Not Fall', (rectangle[0], rectangle[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # #กรณีคิดเฉพาะความสูงของกล่อง bounding box ทำนายการเกิดการหกล้ม
                    # if((h[len(h)-2]/rectangle[3] > 1) and ((cy - y[len(y)-2]) > 70)):
                    #     cv2.putText(orig_frame, "Fall detect", (rectangle[0], rectangle[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #     cv2.rectangle(orig_frame, point1, point2, (0, 0, 255), 2)
                    #     print("fall detection")
                    #     # r = requests.post(url, headers=headers, data={'message': 'Test Fall Detect System'})
                    #     # print(r.text)
                    #
                    # elif ((cy - y[len(y)-2]) > 60):
                    #     cv2.putText(orig_frame, "Fall detect", (rectangle[0], rectangle[1] - 7),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #     print("fall detection")
                    #     # r = requests.post(url, headers=headers, data={'message': 'Test Fall Detect System'})

                    else:
                        cv2.putText(orig_frame, 'Not Fall', (rectangle[0], rectangle[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print("Not Fall")

        cv2.imshow('frame_detect', orig_frame)
        # cv2.imshow('frame', closing)
        cv2.imshow('frame', fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()