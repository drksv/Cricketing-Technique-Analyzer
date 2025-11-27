import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from cricket_pose_utils import analyze_video_vs_ideal

app = Flask(__name__)

# 50MB upload limit
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# CORS
CORS(app, origins=["https://healthtimeout.in"])

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

video_url_env_map = {
    "pull_shot": "PULL_SHOT_URL",
    "cover_drive": "COVER_DRIVE_URL",
    "yorker": "YORKER_URL",
    "bouncer": "BOUNCER_URL"
}

@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
        <h2>🏏 Health Timeout Cricket Technique Analyzer</h2>
        <form method="POST" action="/analyze" enctype="multipart/form-data">
            <label>Select Scenario:</label>
            <select name="scenario">
                <option value="pull_shot">Pull Shot</option>
                <option value="cover_drive">Cover Drive</option>
                <option value="yorker">Yorker</option>
                <option value="bouncer">Bouncer</option>
            </select><br><br>

            <label>Upload Your Video:</label>
            <input type="file" name="video" accept="video/*" required><br><br>

            <input type="submit" value="Analyze">
        </form>
    """)


@app.route("/analyze", methods=["POST"])
def analyze_video():
    # VALIDATION
    if "video" not in request.files or "scenario" not in request.form:
        return "Missing video or scenario", 400

    scenario = request.form["scenario"]
    video_file = request.files["video"]

    # SCENARIO CHECK
    if scenario not in scenario_mapping["batting"] and scenario not in scenario_mapping["bowling"]:
        return "Invalid scenario", 400

    # GET IDEAL URL
    video_env_key = video_url_env_map.get(scenario)
    if not video_env_key:
        return f"No env var mapping for {scenario}", 400

    ideal_url = os.getenv(video_env_key)
    if not ideal_url:
        return f"Environment variable {video_env_key} not set.", 500

    # SAVE UPLOADED VIDEO STREAMING — NO RAM USED
    user_video_temp = os.path.join("/tmp", "user_upload.mp4")
    with open(user_video_temp, "wb") as f:
        for chunk in video_file.stream:
            f.write(chunk)

    # RUN ANALYZER
    result = analyze_video_vs_ideal(user_video_temp, ideal_url)

    return jsonify(result)
