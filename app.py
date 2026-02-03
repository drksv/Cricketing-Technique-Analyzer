import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from cricket_pose_utils import analyze_video_vs_ideal

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Scenario → env var mapping
video_url_env_map = {
    "pull_shot": "PULL_SHOT_URL",
    "cover_drive": "COVER_DRIVE_URL",
    "yorker": "YORKER_URL",
    "bouncer": "BOUNCER_URL",
    "left_arm_spin": "LEFT_ARM_SPIN_URL"
}


@app.route("/analyze", methods=["POST"])
def analyze_video():
    scenario = request.form.get("scenario")
    fatigue = request.form.get("fatigue", "true").lower() == "true"

    if scenario not in video_url_env_map:
        return jsonify({"error": "Invalid scenario"}), 400

    ideal_video_url = os.getenv(video_url_env_map[scenario])

    if not ideal_video_url:
        return jsonify({
            "error": f"Ideal video URL not configured for {scenario}"
        }), 500

    result = analyze_video_vs_ideal(
        request.files["video"],
        ideal_video_url,
        scenario,
        fatigue_enabled=fatigue
    )

    return jsonify(result)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
