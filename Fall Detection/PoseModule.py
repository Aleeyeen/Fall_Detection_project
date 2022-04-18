import cv2
import mediapipe as mp
from datetime import datetime
import os
import pyrebase

# def speedFinder(distance, timeTaken):
#     speed = distance / timeTaken
#     return speed
#
# def averageFinder(completList, averageOfItem):
#     list = len(completList)
#     selectItem = list - averageOfItem
#     selectItemlist = completList[selectItem:]
#     average = sum(selectItemlist) / len(selectItemlist)
#     return average

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                w = img.shape[1]
                h = img.shape[0]
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    date = datetime.today()
    # date.ctime()
    date_ = str(date.strftime("%d/%m/%Y %H:%M:%S"))

    config = {
        "apiKey": "AIzaSyAMQP9fswkwj4Vitb_8MhUBmvq5Zf1LVQA",
        "authDomain": "fall-data.firebaseapp.com",
        "databaseURL": "https://fall-data-default-rtdb.firebaseio.com",
        "projectId": "fall-data",
        "storageBucket": "fall-data.appspot.com",
        "messagingSenderId": "252352858299",
        "appId": "1:252352858299:web:3459d8089879a35767419e",
        "measurementId": "G-6ZRL8QHKVD"
    }

    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    data_1 = {
        "detection": "Fall Detected",
        "date": date_,
        "type": "Caution"
    }
    data_2 = {
        "detection": "Fall Detected",
        "date": date_,
        "type": "Emergency "
    }

    cap = cv2.VideoCapture('Fall2.mp4')
    # path = 'D:\project_II\Fall_log'
    detector = poseDetector()

    y = []
    count = 1

    # initialtime = 0
    # listSpeed = []

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (600, 400))
        if not ret:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            cy = lmList[0]
            y.append(cy)
            r = ((cy - y[len(y)-2]))
            # d = (cy - y[len(y)-2])
            # changeIntime = time.time() - initialtime
            # print(changeIntime)
            # speed = speedFinder(r, changeIntime)
            # listSpeed.append(speed)
            #
            # averageSpeed = averageFinder(listSpeed, 10)
            # if averageSpeed < 0:
            #     averageSpeed *= -1
            #     print(averageSpeed)

            # print(listSpeed)
            # print(y)
            # print("r ", r)
            # print("d", d)

            if(r >= 32.11 ):
                if(r >= 42.13 and r <= 64.15):
                    print("r ", r)
                    print("fall detected Warning")
                        # while count >= 1:
                        #     cv2.imwrite(os.path.join(path, f'Fall_Detected{count}.jpg'), img)
                        #     break
                        # count += 1
                        # database.child('detection').push(data_1)
                else:
                    if( r > 64.15):
                        print("Fall detection Emergency")

            # elif(r < 0):
            #     print('Not Fall')
            # print('r', r)
            # print("cy ", str(cy))

        # cv2.imshow('image', img1)
        cv2.imshow('img_detect', img)
        cv2.waitKey(1)

if __name__ == "__main__": main()