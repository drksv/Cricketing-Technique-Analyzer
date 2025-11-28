# cricket_pose_utils.py
import os
import tempfile
import urllib.request
import cv2
import numpy as np
import logging
from typing import List

# use already-loaded _pose from app import to avoid re-init — but if running standalone, create local Pose
try:
    from app import _pose as GLOBAL_POSE  # when run under Gunicorn with preload
except Exception:
    import mediapipe as mp
    GLOBAL_POSE = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=0, smooth_landmarks=False, enable_segmentation=False
    )

logging.basicConfig(level=logging.INFO)

# Cache for ideal video paths by scenario
IDEAL_CACHE = {}

def download_and_cache_ideal(scenario: str, url: str, max_bytes: int = 25 * 1024 * 1024) -> str:
    """
    Download ideal video once per instance and cache on disk.
    Returns local path.
    """
    if scenario in IDEAL_CACHE and os.path.exists(IDEAL_CACHE[scenario]):
        return IDEAL_CACHE[scenario]

    tmp_path = os.path.join(tempfile.gettempdir(), f"ideal_{scenario}.mp4")
    if os.path.exists(tmp_path):
        IDEAL_CACHE[scenario] = tmp_path
        return tmp_path

    logging.info(f"Downloading ideal video for {scenario} -> {tmp_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "healthtimeout-cricket-analyzer/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        total = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 64)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    os.remove(tmp_path)
                    raise ValueError("Ideal video too large")
                f.write(chunk)

    IDEAL_CACHE[scenario] = tmp_path
    return tmp_path

# parameters tuned for low-ram environments
FRAME_SKIP = 4          # process 1 in every 4 frames
MAX_PROCESSED_FRAMES = 120
RESIZE_DIMS = (320, 180)  # low-res frames

def extract_landmarks_from_video_path(video_path: str) -> List[List[tuple]]:
    """
    Extract simplified landmarks (x,y,z,visibility) per processed frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Cannot open video: {video_path}")
        return []

    landmarks_seq = []
    frame_i = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if frame_i % FRAME_SKIP != 0:
            continue

        # safety cap
        if processed >= MAX_PROCESSED_FRAMES:
            break
        processed += 1

        try:
            small = cv2.resize(frame, RESIZE_DIMS)
        except Exception:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = GLOBAL_POSE.process(rgb)
        if results and results.pose_landmarks:
            lm = [(p.x, p.y, getattr(p, "z", 0.0), getattr(p, "visibility", 0.0)) for p in results.pose_landmarks.landmark]
            landmarks_seq.append(lm)

    cap.release()
    return landmarks_seq

def compare_sequences(user_seq, ideal_seq):
    if not user_seq or not ideal_seq:
        return {"score": 0, "issues": ["Could not extract poses from one or both videos."]}

    n = min(len(user_seq), len(ideal_seq))
    diffs = []
    issues = []

    for i in range(n):
        u = np.array(user_seq[i])[:, :3]
        v = np.array(ideal_seq[i])[:, :3]
        # align based on number of landmarks; if mismatch skip
        if u.shape != v.shape:
            continue
        dist = np.linalg.norm(u - v, axis=1)
        diffs.append(np.mean(dist))
        # Key joint checks (indices are from MediaPipe pose landmarks)
        key_joints = {"Shoulder": 11, "Elbow": 13, "Knee": 25, "Ankle": 27}
        for name, idx in key_joints.items():
            if idx < u.shape[0]:
                if dist[idx] > 0.08:
                    issues.append(f"Frame {i+1}: {name} misaligned (diff {dist[idx]:.3f})")

    if not diffs:
        return {"score": 0, "issues": ["No comparable frames found."]}

    avg_diff = float(np.mean(diffs))
    score = max(0.0, min(100.0, 100.0 - (avg_diff * 100.0)))  # heuristic
    if not issues:
        issues = ["Looks good — minimal adjustments needed."]

    return {"score": round(score, 2), "issues": issues, "avg_frame_diff": round(avg_diff, 4)}

# High-level functions used by app.py
def perform_analysis(user_video_path: str, ideal_video_path: str):
    user_seq = extract_landmarks_from_video_path(user_video_path)
    ideal_seq = extract_landmarks_from_video_path(ideal_video_path)
    return compare_sequences(user_seq, ideal_seq)

# expose download function for app.py import usage
download_and_cache_ideal = download_and_cache_ideal
