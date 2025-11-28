from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import shutil

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

# Folders for uploads and processed videos
UPLOAD_FOLDER = "/tmp/uploads"
PROCESSED_FOLDER = "/tmp/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Dummy analysis function (replace with your cricket pose analysis)
def run_cricket_analysis(input_path):
    filename = os.path.basename(input_path)
    processed_name = f"processed_{uuid.uuid4().hex}_{filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
    # Copy file as placeholder for real analysis
    shutil.copy(input_path, processed_path)
    return processed_name

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_video' not in request.files:
        return jsonify({'error': 'Missing user_video'}), 400

    video_file = request.files['user_video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded video
    saved_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(saved_path)

    # Run analysis
    processed_filename = run_cricket_analysis(saved_path)

    # Generate full URL for frontend (replace with your actual deployed domain)
    analysis_url = f"https://cricketing-technique-analyzer.onrender.com/processed/{processed_filename}"

    return jsonify({
        'status': 'Analysis complete!',
        'analysis_video_url': analysis_url
    })

# Serve processed videos
@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
