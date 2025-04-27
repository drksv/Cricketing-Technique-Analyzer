import os
import json
from flask import Flask, request, jsonify, render_template_string

from cricket_pose_utils import analyze_video_vs_ideal  # Make sure you import your function properly

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload limit

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

@app.route("/", methods=["GET"])
def home():
    # Simple HTML form
    return render_template_string('''
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
    ''')

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if 'video' not in request.files or 'scenario' not in request.form:
        return "Missing video file or scenario", 400

    video_file = request.files['video']
    scenario = request.form['scenario']

    if scenario in scenario_mapping['batting'] or scenario in scenario_mapping['bowling']:
        video_env_var = video_url_env_map.get(scenario)
        if not video_env_var:
            return "Video URL for scenario not configured.", 400

        ideal_video_url = os.getenv(video_env_var)
        if not ideal_video_url:
            return f"Environment variable {video_env_var} not set.", 500
    else:
        return "Invalid scenario selected.", 400

    result = analyze_video_vs_ideal(video_file, ideal_video_url)
    return jsonify(result)
