import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import tempfile
import os

mp_pose = mp.solutions.pose

def download_video(url, save_path):
    """Download video from a URL to a local path."""
    urllib.request.urlretrieve(url, save_path)

def extract_landmarks_from_video(video_path):
    """Extract pose landmarks from a video file."""
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image color and process
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)

    cap.release()
    return landmarks_list

def compare_poses(user_landmarks, ideal_landmarks):
    """Compare user video landmarks vs ideal video landmarks and score."""
    frame_count = min(len(user_landmarks), len(ideal_landmarks))

    if frame_count == 0:
        return 0, ["Could not detect poses in one or both videos. Ensure both videos show the full body clearly."]

    total_score = 0
    issues = []

    for i in range(frame_count):
        user_frame = user_landmarks[i]
        ideal_frame = ideal_landmarks[i]
        frame_score = 0

        for idx, (u, ideal) in enumerate(zip(user_frame, ideal_frame)):
            dist = np.linalg.norm(np.array(u) - np.array(ideal))
            frame_score += max(0, 1 - dist * 10)

        frame_score /= len(user_frame)
        total_score += frame_score * 100  # Scale to 100

        key_joints = {"Elbow": 13, "Knee": 25, "Shoulder": 11, "Ankle": 27}
        for name, idx in key_joints.items():
            dist = np.linalg.norm(np.array(user_frame[idx]) - np.array(ideal_frame[idx]))
            if dist > 0.08:
                issues.append(f"Frame {i+1}: {name} needs improvement.")

    final_score = max(0, min(100, total_score / frame_count))
    return final_score, issues


def analyze_video_vs_ideal(user_video_file, ideal_video_url):
    """Main analysis function."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        user_video_path = tmp.name
        user_video_file.save(user_video_path)

    ideal_video_path = os.path.join(tempfile.gettempdir(), "ideal_video.mp4")
    download_video(ideal_video_url, ideal_video_path)

    user_landmarks = extract_landmarks_from_video(user_video_path)
    ideal_landmarks = extract_landmarks_from_video(ideal_video_path)

    os.remove(user_video_path)
    os.remove(ideal_video_path)

    if not user_landmarks:
        return {"score": 0, "issues": ["Could not detect pose in your video. Make sure the body is clearly visible."]}

    if not ideal_landmarks:
        return {"score": 0, "issues": ["Could not process the ideal video. Check configuration."]}

    score, issues = compare_poses(user_landmarks, ideal_landmarks)

    result = {
        "score": round(score, 2),
        "issues": issues if issues else ["Looks great! Minimal adjustments needed."]
    }
    return result
