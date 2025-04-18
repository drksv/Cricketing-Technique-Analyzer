import os
from flask import Flask, request, jsonify
from cricket_pose_utils import analyze_video_vs_ideal
import json

app = Flask(__name__)

# Load scenario mapping from JSON
with open('utils/scenario_mapping.json') as f:
    scenario_mapping = json.load(f)

# Root route — health check / homepage
@app.route('/')
def home():
    return "🏏 Hello from Health Timeout Cricket Technique Analyzer!"

# Route to get available scenarios
@app.route('/scenarios', methods=['GET'])
def get_scenarios():
    return jsonify(scenario_mapping)

# Video analysis route
@app.route('/analyze', methods=['POST'])
def analyze_video():
    video = request.files.get('video')
    scenario = request.form.get('scenario')

    if not video or not scenario:
        return jsonify({'error': 'Please provide both video and scenario.'}), 400

    video_path = 'temp_user_video.mp4'
    video.save(video_path)

    ideal_video_url = scenario_mapping.get('batting' if scenario in scenario_mapping['batting'] else 'bowling', {}).get(scenario)

    if not ideal_video_url:
        return jsonify({'error': f'Scenario {scenario} not found.'}), 404

    score, issues = analyze_video_vs_ideal(video_path, ideal_video_url)

    return jsonify({
        'score': score,
        'issues': issues
    })

if __name__ == '__main__':
    # Pick up port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
