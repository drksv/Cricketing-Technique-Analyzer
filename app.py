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

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_video' not in request.files:
        return jsonify({'error': 'Missing user_video'}), 400

    video_file = request.files['user_video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save temporarily
    temp_path = f"/tmp/{video_file.filename}"
    video_file.save(temp_path)

    # Run your cricket analysis here
    # For now, return placeholder response
    return jsonify({
        'status': 'Video uploaded successfully!',
        'analysis_video_url': None  # replace with actual processed video URL if available
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

