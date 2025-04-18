import os
import json
from flask import Flask, request

app = Flask(__name__)

# Load scenario mapping
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

# Map scenario names to environment variable keys
video_url_env_map = {
    "pull_shot": "PULL_SHOT_URL",
    "cover_drive": "COVER_DRIVE_URL",
    "yorker": "YORKER_URL",
    "bouncer": "BOUNCER_URL"
}

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if 'video' not in request.files or 'scenario' not in request.form:
        return "Missing video file or scenario", 400

    video_file = request.files['video']
    scenario = request.form['scenario']

    # Check if scenario is valid
    if scenario in scenario_mapping['batting'] or scenario in scenario_mapping['bowling']:
        video_env_var = video_url_env_map.get(scenario)
        if not video_env_var:
            return "Video URL for scenario not configured.", 400

        ideal_video_url = os.getenv(video_env_var)
        if not ideal_video_url:
            return f"Environment variable {video_env_var} not set.", 500
    else:
        return "Invalid scenario selected.", 400

    # Now use ideal_video_url in your pose analysis function
    result = analyze_video_vs_ideal(video_file, ideal_video_url)
    return result
