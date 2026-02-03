import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import tempfile
import os

mp_pose = mp.solutions.pose

# -----------------------------
# Scenario-specific key joints
# -----------------------------
SCENARIO_JOINTS = {

    # ---------- BOWLING ----------
    "left_arm_spin": {
        "Bowling Arm": [11, 13, 15],
        "Shoulders": [11, 12],
        "Hips": [23, 24],
        "Front Knee": [25],
        "Ankles": [27, 28]
    },

    "yorker": {
        "Bowling Arm": [11, 13, 15],
        "Front Knee": [25],
        "Spine": [11, 12, 23, 24]
    },

    "bouncer": {
        "Bowling Arm": [11, 13, 15],
        "Shoulders": [11, 12],
        "Back Extension": [11, 12, 23, 24],
        "Front Leg Bracing": [25],
        "Landing Ankle": [27]
    },

    # ---------- BATTING ----------
    "pull_shot": {
        "Batting Arm": [12, 14, 16],
        "Head Position": [0],
        "Hip Rotation": [23, 24],
        "Back Knee": [26],
        "Balance": [27, 28]
    },

    "cover_drive": {
        "Batting Arm": [12, 14, 16],
        "Front Knee Bend": [25],
        "Shoulder Alignment": [11, 12],
        "Head Over Ball": [0],
        "Foot Placement": [27, 28]
    }
}


def extract_landmarks_from_video(video_path, frame_skip=5, max_frames=120):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=False
    )

    landmarks = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        frame = cv2.resize(frame, (320, 180))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks.append([(l.x, l.y) for l in result.pose_landmarks.landmark])

        if len(landmarks) >= max_frames:
            break

    cap.release()
    return landmarks


def compare_poses(user_landmarks, ideal_landmarks, scenario):
    frame_count = min(len(user_landmarks), len(ideal_landmarks))
    if frame_count == 0:
        return 0, ["Pose not detected clearly."]

    joints_map = SCENARIO_JOINTS.get(
        scenario,
        {"General Alignment": [11, 12, 23, 24]}
    )

    total_score = 0
    issues = []

    for i in range(frame_count):
        frame_score = 0
        for part, indices in joints_map.items():
            diffs = []
            for idx in indices:
                dist = np.linalg.norm(
                    np.array(user_landmarks[i][idx]) -
                    np.array(ideal_landmarks[i][idx])
                )
                diffs.append(dist)

            part_score = max(0, 1 - np.mean(diffs) * 8)
            frame_score += part_score

            if np.mean(diffs) > 0.08:
                issues.append(f"{part} deviation detected (frame {i+1})")

        frame_score /= len(joints_map)
        total_score += frame_score * 100

    final_score = min(100, total_score / frame_count)
    return round(final_score, 2), list(set(issues))


def detect_fatigue(user_landmarks, scenario):
    """
    Detects fatigue by checking posture deviation trend over time.
    """
    if len(user_landmarks) < 10:
        return False, "Not enough frames for fatigue analysis."

    joints = SCENARIO_JOINTS.get(scenario)
    if not joints:
        return False, "Fatigue model not available for this technique."

    early = user_landmarks[:len(user_landmarks)//3]
    late = user_landmarks[-len(user_landmarks)//3:]

    def avg_deviation(frames):
        deviations = []
        for frame in frames:
            for indices in joints.values():
                for idx in indices:
                    deviations.append(np.linalg.norm(np.array(frame[idx])))
        return np.mean(deviations)

    early_dev = avg_deviation(early)
    late_dev = avg_deviation(late)

    fatigue_score = late_dev - early_dev

    if fatigue_score > 0.03:
        return True, "Posture consistency dropped under fatigue."

    return False, "No significant fatigue-related breakdown detected."


def analyze_video_vs_ideal(user_video_file, ideal_video_url, scenario, fatigue_enabled=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        user_video_path = tmp.name
        user_video_file.save(user_video_path)

    ideal_video_path = os.path.join(tempfile.gettempdir(), "ideal_video.mp4")
    urllib.request.urlretrieve(ideal_video_url, ideal_video_path)

    user_landmarks = extract_landmarks_from_video(user_video_path)
    ideal_landmarks = extract_landmarks_from_video(ideal_video_path)

    os.remove(user_video_path)
    os.remove(ideal_video_path)

    score, issues = compare_poses(user_landmarks, ideal_landmarks, scenario)

    fatigue_result = None
    if fatigue_enabled:
        fatigue_result = detect_fatigue(user_landmarks, scenario)

    return {
        "score": score,
        "issues": issues if issues else ["Technique looks solid."],
        "fatigue": fatigue_result
    }
