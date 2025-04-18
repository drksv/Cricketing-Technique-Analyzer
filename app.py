import os
import json
from flask import Flask, request, jsonify
from cricket_pose_utils import analyze_video_vs_ideal

app = Flask(__name__)

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

# API endpoint to check available scenarios
@app.route("/scenarios", methods=["GET"])
def get_scenarios():
    return jsonify(scenario_mapping)

# Function to get video URL from env vars
def get_video_url(scenario_option):
    mapping = {
        "pull_shot": os.environ.get("PULL_SHOT_URL"),
        "cover_drive": os.environ.get("COVER_DRIVE_URL"),
        "yorker": os.environ.get("YORKER_URL"),
        "bouncer": os.environ.get("BOUNCER_URL")
    }
    return mapping.get(scenario_option)

# API endpoint to analyze uploaded video
@app.route("/analyze", methods=["POST"])
def analyze_video():
    scenario_option = request.form.get("scenario")
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    video_path = "temp_user_video.mp4"
    video_file.save(video_path)

    ideal_video_url = get_video_url(scenario_option)
    if not ideal_video_url:
        return jsonify({"error": "Invalid scenario selected"}), 400

    result = analyze_video_vs_ideal(video_path, ideal_video_url)

    os.remove(video_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
