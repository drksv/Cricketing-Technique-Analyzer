# app.py
import os
import tempfile
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# quiet tensorflow/mediapipe logs a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2  # keep after threads env
cv2.setNumThreads(1)

# preload mediapipe model at import time (prevents per-request downloads)
import mediapipe as mp
mp_pose = mp.solutions.pose
_pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False
)

logging.basicConfig(level=logging.INFO)
logging.info("App starting — MediaPipe pose model preloaded.")

# Flask app
app = Flask(__name__)
# Safety upload limit (50MB). Adjust down if needed.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Allow your site(s) to call API — adjust for production
CORS(app, origins=[
    "https://healthtimeout.in",
    "https://www.healthtimeout.in",
    "http://localhost:3000",
    "http://localhost"
])

# Scenario mapping file (optional) — keep a JSON in utils/
SCENARIO_MAP_PATH = "utils/scenario_mapping.json"
if os.path.exists(SCENARIO_MAP_PATH):
    with open(SCENARIO_MAP_PATH, "r") as f:
        scenario_mapping = json.load(f)
else:
    # fallback structure if file missing
    scenario_mapping = {"batting": ["pull_shot", "cover_drive"], "bowling": ["yorker", "bouncer"]}

# Map scenario -> env var name for ideal video URL
VIDEO_ENV_MAP = {
    "pull_shot": "PULL_SHOT_URL",
    "cover_drive": "COVER_DRIVE_URL",
    "yorker": "YORKER_URL",
    "bouncer": "BOUNCER_URL"
}

# import analyzer function (uses preloaded _pose from above)
from cricket_pose_utils import analyze_video_vs_ideal_cached  # defined in cricket_pose_utils.py

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "service": "cricket-analyzer"})

@app.route("/analyze", methods=["GET"])
def analyze_get():
    return jsonify({"error": "POST multipart/form-data with fields: scenario, video (file)"}), 405

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        # basic checks
        if "video" not in request.files or "scenario" not in request.form:
            return jsonify({"error": "Missing 'video' file or 'scenario' field"}), 400

        video_file = request.files["video"]
        scenario = request.form["scenario"]

        # validate scenario
        if (scenario not in scenario_mapping.get("batting", []) and
                scenario not in scenario_mapping.get("bowling", [])):
            return jsonify({"error": "Invalid scenario"}), 400

        env_key = VIDEO_ENV_MAP.get(scenario)
        if not env_key:
            return jsonify({"error": f"No env var mapping for scenario {scenario}"}), 500

        ideal_url = os.getenv(env_key)
        if not ideal_url:
            return jsonify({"error": f"Ideal video URL not set for {env_key} in environment"}), 500

        # quick size check (raise 413 if too big)
        video_file.seek(0, os.SEEK_END)
        size_mb = video_file.tell() / (1024 * 1024)
        video_file.seek(0)
        if size_mb > 45:  # keep below app MAX_CONTENT_LENGTH
            return jsonify({"error": "Uploaded file too large (max 45 MB)"}), 413

        # save user upload to /tmp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf_user:
            # write in streaming chunks to avoid buffering huge memory
            chunk = video_file.stream.read(1024 * 64)
            while chunk:
                tf_user.write(chunk)
                chunk = video_file.stream.read(1024 * 64)
            user_path = tf_user.name

        # Get ideal video path via cached downloader in cricket_pose_utils
        ideal_path = analyze_video_vs_ideal_cached.download_and_cache_ideal(scenario, ideal_url)

        # Run analysis (this function returns small JSON-able dict)
        result = analyze_video_vs_ideal_cached.perform_analysis(user_path, ideal_path)

        # cleanup user file
        try:
            os.remove(user_path)
        except Exception:
            pass

        return jsonify(result)

    except Exception as e:
        logging.exception("Error in /analyze")
        return jsonify({"error": "Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
