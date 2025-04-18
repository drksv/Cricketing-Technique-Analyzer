import cv2
import mediapipe as mp
import urllib.request
import numpy as np

mp_pose = mp.solutions.pose

def download_video(video_url, save_path):
    urllib.request.urlretrieve(video_url, save_path)

def extract_pose_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    landmarks_all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_all_frames.append(landmarks)
    cap.release()
    return landmarks_all_frames

def compare_poses(user_landmarks, ideal_landmarks):
    frame_count = min(len(user_landmarks), len(ideal_landmarks))
    total_score = 0
    issues = []

    important_joints = {
        0: 'nose', 11: 'left_shoulder', 12: 'right_shoulder', 
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle'
    }

    for i in range(frame_count):
        user_frame = user_landmarks[i]
        ideal_frame = ideal_landmarks[i]
        frame_score = 0

        for idx, joint in important_joints.items():
            user_joint = np.array(user_frame[idx])
            ideal_joint = np.array(ideal_frame[idx])
            distance = np.linalg.norm(user_joint - ideal_joint)
            if distance < 0.05:
                frame_score += 1
            else:
                issues.append(f"{joint.replace('_', ' ').title()} needs adjustment in frame {i+1}")

        total_score += (frame_score / len(important_joints)) * 100

    final_score = max(0, min(100, total_score / frame_count)) if frame_count > 0 else 0
    return final_score, issues

def analyze_video_vs_ideal(user_video_path, ideal_video_url):
    download_video(ideal_video_url, "temp_ideal_video.mp4")
    user_landmarks = extract_pose_landmarks(user_video_path)
    ideal_landmarks = extract_pose_landmarks("temp_ideal_video.mp4")
    score, issues = compare_poses(user_landmarks, ideal_landmarks)
    return {"score": round(score, 2), "issues": issues}
