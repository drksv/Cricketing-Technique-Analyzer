import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )

    def analyze_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if not result.pose_landmarks:
            return {"status": "no-person"}

        lm = result.pose_landmarks.landmark

        left_elbow = lm[13]
        left_shoulder = lm[11]
        left_wrist = lm[15]

        angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        return {
            "status": "ok",
            "elbow_angle": angle
        }

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(cosine_angle))

        return round(angle, 2)
