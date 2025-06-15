# HandDetectionModule.py
import cv2
import mediapipe as mp

print("HandDetectionModule loaded successfully.")

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.handType = []
        self.bboxCenter = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findLandMark(self, img, handNo=0, draw=True):
        lmList = []
        self.handType = []
        self.bboxCenter = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList, yList = [], []
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Get hand label: "Left" or "Right"
            if self.results.multi_handedness:
                handedness = self.results.multi_handedness[handNo]
                self.handType.append(handedness.classification[0].label)

            # Store bounding box center
            x_min, x_max = min(xList), max(xList)
            y_min, y_max = min(yList), max(yList)
            self.bboxCenter.append(((x_min + x_max) // 2, (y_min + y_max) // 2))

        return lmList
