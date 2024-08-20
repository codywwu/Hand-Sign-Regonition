import cv2
import numpy as np
import math

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
imgSize = 300
offset = 20
counter = 0
labels = ["Middle","P"]

labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
folder = "images/C"


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.findHands(img,draw = False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x- offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize /h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize
                                             , hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw =  False)

        #"F" word dection 
        if(labels[index] == 'B' ):
            cv2.putText(img, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), -10)
        else:

            cv2.putText(img, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), )

    cv2.imshow("Image",img)
    cv2.waitKey(1)

