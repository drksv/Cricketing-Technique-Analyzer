import os
import json
import tempfile
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS

from cricket_pose_utils import analyze_video_vs_ideal  # Your pose analysis functions

app = Flask(__name__)
CORS(app)  # Allow all origins for simplicity

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Mapping scenario → environment variable storing ideal video URL
video_url_env_map = {
    "pull_shot": "PULL_SHOT_URL",
    "cover_drive": "COVER_DRIVE_URL",
    "yorker": "YORKER_URL",
    "bouncer": "BOUNCER_URL"
}

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if 'video' not in request.files or 'scenario' not in request.form:
        return jsonify({"error": "Missing video or scenario"}), 400

    user_video_file = request.files['video']
    scenario = request.form['scenario']

    if scenario not in video_url_env_map:
        return jsonify({"error": "Invalid scenario selected"}), 400

    ideal_video_url = os.getenv(video_url_env_map[scenario])
    if not ideal_video_url:
        return jsonify({"error": f"Ideal video URL for {scenario} not configured"}), 500

    try:
        result = analyze_video_vs_ideal(user_video_file, ideal_video_url)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": "Failed to analyze video"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
