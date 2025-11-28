import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------------
# 1. Load MediaPipe Pose ONCE (major optimization)
# ------------------------------------------------------------
mp_pose = mp.solutions.pose
_pose = mp_pose.Pose(
    static_image_mode=True,        # Faster for frame-by-frame
    model_complexity=0,            # Lite model — required for low memory
    smooth_landmarks=False,
    enable_segmentation=False
)

# ------------------------------------------------------------
# 2. Extract pose landmarks from video (optimized)
# ------------------------------------------------------------
def extract_landmarks_from_video(video_path, frame_skip=5, max_frames=150):
    """
    Extract pose landmarks from a video by sampling every Nth frame.
    - frame_skip=5 → processes every 6th frame → huge speed boost
    - max_frames caps usage so Render workers don't timeout
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    all_frames_landmarks = []
    frame_count = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed & memory
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame_count += 1
            processed_frames += 1

            # Safety cap → prevents Render timeouts
            if processed_frames > max_frames:
                break

            # Downscale frame to reduce MediaPipe load
            frame = cv2.resize(frame, (320, 180))  # lightweight resolution

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run pose detection
            results = _pose.process(rgb)

            # Save landmarks (if detected)
            if results.pose_landmarks:
                lm = [
                    (
                        lm.x,
                        lm.y,
                        lm.z,
                        lm.visibility
                    )
                    for lm in results.pose_landmarks.landmark
                ]
                all_frames_landmarks.append(lm)

    finally:
        cap.release()

    return all_frames_landmarks


# ------------------------------------------------------------
# 3. Compare user vs ideal pose sequences
# ------------------------------------------------------------
def compare_landmark_sequences(user_seq, ideal_seq):
    """
    Compute a simple difference score between two landmark sequences.
    The shorter one determines the number of frames used.
    """

    if len(user_seq) == 0 or len(ideal_seq) == 0:
        return {
            "score": 0,
            "feedback": "Unable to extract pose landmarks from one or both videos."
        }

    n = min(len(user_seq), len(ideal_seq))
    diffs = []

    for i in range(n):
        u = np.array(user_seq[i])
        v = np.array(ideal_seq[i])

        # Euclidean difference across landmarks
        diff = np.linalg.norm(u[:, :3] - v[:, :3])
        diffs.append(diff)

    avg_diff = float(np.mean(diffs))

    # Lower diff → better similarity
    score = max(0, 100 - avg_diff * 10)

    return {
        "score": round(score, 2),
        "avg_difference": round(avg_diff, 4)
    }


# ------------------------------------------------------------
# 4. High-level API called by Flask endpoint
# ------------------------------------------------------------
def analyze_video_vs_ideal(user_video_path, ideal_video_path):
    print("[INFO] Extracting user video pose...")
    user_lm = extract_landmarks_from_video(user_video_path)

    print("[INFO] Extracting ideal video pose...")
    ideal_lm = extract_landmarks_from_video(ideal_video_path)

    print("[INFO] Comparing pose frames...")
    result = compare_landmark_sequences(user_lm, ideal_lm)

    return result
