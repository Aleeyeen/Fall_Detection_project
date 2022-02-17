import cv2
import mediapipe as mp
import time

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
                cx, cy = (lm.x * w), (lm.y * h)
                lmList.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

# def main():
#     cap = cv2.VideoCapture(0)
#     detector = poseDetector()
#     y = []
#     prev_frame_time = 0
#     new_frame_time = 0
#     while True:
#         ret, img = cap.read()
#         img1 = img.copy()
#         if not ret:
#             break

#         # font = cv2.FONT_HERSHEY_SIMPLEX
#         # new_frame_time = time.time()

#         # fps = 1/(new_frame_time - prev_frame_time)
#         # prev_frame_time = new_frame_time
#         # fps = int(fps)
#         # #print(fps)

#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             cy = lmList[0]
#             y.append(cy)
#             r = (cy - y[len(y)-2])
#             if ((r) > 44.10):
#                 print("fall detection")
#             # print(y)
#             print('r ',r)
#             # print("cy ", str(cy))

#         #cv2.imshow('image', img1)
#         cv2.imshow('img_detect', img)
#         cv2.waitKey(1)

# if __name__ == "__main__": main()