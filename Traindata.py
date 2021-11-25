import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np


cap = cv2.VideoCapture(0)

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

frames = []

model = tensorflow.keras.models.load_model('keras_model.h5')
size = (224, 224)

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

frames_zr = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_frame = cv2.absdiff(gray, grayMedianFrame)
        _, fgmask = cv2.threshold(diff_frame, 90, 255, cv2.THRESH_BINARY)

        kernel = np.ones((10, 10), np.uint8)
        fgmask = cv2.dilate(fgmask, kernel, iterations=5)
        fgmask = cv2.erode(fgmask, kernel, iterations=2)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=6)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Area = 0

        for c in contours:
            vertices = cv2.boundingRect(c)
            if(cv2.contourArea(c) > Area):
                Area = cv2.contourArea(c)
                rectangle = vertices

                point1 = (rectangle[0], rectangle[1])
                point2 = (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
                cv2.rectangle(orig_frame, point1, point2, (0, 255, 0), 1)

                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                crop = Image.fromarray(orig_frame[rectangle[1]: rectangle[1] + rectangle[3], rectangle[0]: rectangle[0] + rectangle[2]])
                image = crop
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                print(prediction)

                if prediction[0][0] < prediction[0][1]:
                    cv2.putText(orig_frame, 'Fall', (rectangle[0], rectangle[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(orig_frame, point1, point2, (0, 0, 255), 1)
                    #print("fall")
                else:
                    cv2.putText(orig_frame, 'Not Fall', (rectangle[0], rectangle[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #print("Not Fall")

        cv2.imshow('frame_detect', orig_frame)
        # cv2.imshow('frame_mask', fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()