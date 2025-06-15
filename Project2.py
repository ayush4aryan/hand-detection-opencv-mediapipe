import os
import time
import cv2
import mediapipe as mp
import numpy as np
import HandDetectionModule as HDM

wCam, hCam = 640, 480
tipIds = [4, 8, 12, 16, 20]

detector = HDM.handDetector(detectionCon=0.75)
cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findLandMark(img, draw=False)

    if len(lmList) != 0 and len(detector.handType) != 0:
        fingersUp = []

        # Adjust thumb logic for handedness
        if detector.handType[0] == "Right":
            fingersUp.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
        elif detector.handType[0] == "Left":
            fingersUp.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] else 0)

        # Other 4 fingers
        for id in range(1, 5):
            fingersUp.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

        totalFingers = fingersUp.count(1)

        # Draw finger count
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), -1)
        cv2.putText(img, str(totalFingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        # Hand type label (Left/Right)
        handLabel = detector.handType[0]
        handPos = detector.bboxCenter[0]
        cv2.putText(img, handLabel, (handPos[0] - 40, handPos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Gesture recognition
        gesture = "Unknown"
        if fingersUp == [0, 1, 1, 0, 0]:
            gesture = "Peace "
        elif fingersUp == [1, 0, 0, 0, 0]:
            gesture = "Thumbs Up 👍"
        elif fingersUp == [0, 0, 0, 0, 0]:
            gesture = "Fist ✊"
        elif fingersUp == [1, 1, 1, 1, 1]:
            gesture = "Open Palm 🖐️"

        cv2.putText(img, f'Gesture: {gesture}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
