import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

from cricket_pose_utils import analyze_video_vs_ideal

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return {"status": "Cricket Pose Analyzer is live!"}

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        if "user_video" not in request.files:
            return jsonify({"error": "Missing user_video"}), 400

        ideal_url = request.form.get("ideal_url")
        if not ideal_url:
            return jsonify({"error": "Missing ideal_url"}), 400

        user_video = request.files["user_video"]

        # --- SAFETY LIMIT: reject > 30 MB files ---
        user_video.seek(0, os.SEEK_END)
        size_mb = user_video.tell() / (1024 * 1024)
        if size_mb > 30:
            return jsonify({"error": "User video too large (max 30 MB)"}), 413
        user_video.seek(0)

        # Save user video
        t_user = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        user_video.save(t_user.name)

        # Download ideal video
        t_ideal = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        r = requests.get(ideal_url, stream=True, timeout=5)
        if r.status_code != 200:
            return jsonify({"error": "Failed downloading ideal video"}), 400

        downloaded = 0
        for chunk in r.iter_content(1024 * 64):
            downloaded += len(chunk)
            if downloaded > (25 * 1024 * 1024):     # 25MB limit
                return jsonify({"error": "Ideal video too large"}), 413
            t_ideal.write(chunk)
        t_ideal.flush()

        # Run analysis
        result = analyze_video_vs_ideal(t_user.name, t_ideal.name)

        # Cleanup
        os.remove(t_user.name)
        os.remove(t_ideal.name)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
