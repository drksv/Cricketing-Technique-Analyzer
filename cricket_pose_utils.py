import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import tempfile
import os

mp_pose = mp.solutions.pose

# Reduce MediaPipe threading → avoids OOM
cv2.setNumThreads(1)

def download_video(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(model_complexity=0)  # much smaller model
    landmarks_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 270))  # reduce memory drastically
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)

    cap.release()
    return landmarks_list


def compare_poses(user_landmarks, ideal_landmarks):
    frame_count = min(len(user_landmarks), len(ideal_landmarks))
    if frame_count == 0:
        return 0, ["No pose detected in one or both videos."]

    issues = []
    total_score = 0

    for i in range(frame_count):
        u = np.array(user_landmarks[i])
        v = np.array(ideal_landmarks[i])

        dist = np.linalg.norm(u - v, axis=1)
        frame_score = 1 - np.mean(dist * 10)
        total_score += max(0, frame_score)

        key = {"Elbow": 13, "Knee": 25, "Shoulder": 11, "Ankle": 27}
        for name, idx in key.items():
            if dist[idx] > 0.08:
                issues.append(f"Frame {i+1}: {name} is misaligned.")

    final_score = round(max(0, min(100, total_score / frame_count * 100)), 2)
    return final_score, issues


def analyze_video_vs_ideal(user_video_path, ideal_video_url):
    ideal_path = os.path.join(tempfile.gettempdir(), "ideal_video.mp4")
    download_video(ideal_video_url, ideal_path)

    user_lm = extract_landmarks_from_video(user_video_path)
    ideal_lm = extract_landmarks_from_video(ideal_path)

    if not user_lm:
        return {"score": 0, "issues": ["Pose not detected in your video."]}

    if not ideal_lm:
        return {"score": 0, "issues": ["Ideal video could not be processed."]}

    score, issues = compare_poses(user_lm, ideal_lm)
    return {"score": score, "issues": issues or ["Looks good!"]}
