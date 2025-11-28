import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import tempfile
import os

mp_pose = mp.solutions.pose

# ------------------------------------------------------------
# Extract landmarks from video file path
# ------------------------------------------------------------
def extract_landmarks_from_video(video_path, frame_skip=5, max_frames=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    pose = mp_pose.Pose(static_image_mode=True, model_complexity=0, smooth_landmarks=False)
    landmarks_list = []

    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1
        processed_frames += 1
        if processed_frames > max_frames:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (320, 180))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = [(l.x, l.y) for l in results.pose_landmarks.landmark]
            landmarks_list.append(lm)

    cap.release()
    return landmarks_list

# ------------------------------------------------------------
# Compare user vs ideal landmarks and return score + issues
# ------------------------------------------------------------
def compare_poses(user_landmarks, ideal_landmarks):
    frame_count = min(len(user_landmarks), len(ideal_landmarks))
    if frame_count == 0:
        return 0, ["Could not detect poses in one or both videos. Ensure full body is visible."]

    total_score = 0
    issues = []

    key_joints = {"Elbow": 13, "Knee": 25, "Shoulder": 11, "Ankle": 27}

    for i in range(frame_count):
        user_frame = user_landmarks[i]
        ideal_frame = ideal_landmarks[i]

        frame_score = 0
        for u, ideal in zip(user_frame, ideal_frame):
            dist = np.linalg.norm(np.array(u) - np.array(ideal))
            frame_score += max(0, 1 - dist * 10)
        frame_score /= len(user_frame)
        total_score += frame_score * 100

        for name, idx in key_joints.items():
            dist = np.linalg.norm(np.array(user_frame[idx]) - np.array(ideal_frame[idx]))
            if dist > 0.08:
                issues.append(f"Frame {i+1}: {name} needs improvement.")

    final_score = max(0, min(100, total_score / frame_count))
    return final_score, issues

# ------------------------------------------------------------
# Main API function called from Flask
# ------------------------------------------------------------
def analyze_video_vs_ideal(user_video_file, ideal_video_url):
    # Save user video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        user_video_path = tmp.name
        user_video_file.save(user_video_path)

    # Download ideal video to temp file
    ideal_video_path = os.path.join(tempfile.gettempdir(), "ideal_video.mp4")
    urllib.request.urlretrieve(ideal_video_url, ideal_video_path)

    # Extract landmarks
    user_landmarks = extract_landmarks_from_video(user_video_path)
    ideal_landmarks = extract_landmarks_from_video(ideal_video_path)

    # Clean up temp files
    os.remove(user_video_path)
    os.remove(ideal_video_path)

    if not user_landmarks:
        return {"score": 0, "issues": ["Could not detect pose in your video. Make sure the body is clearly visible."]}
    if not ideal_landmarks:
        return {"score": 0, "issues": ["Could not process the ideal video. Check configuration."]}

    score, issues = compare_poses(user_landmarks, ideal_landmarks)
    return {"score": round(score, 2), "issues": issues if issues else ["Looks great! Minimal adjustments needed."]}
