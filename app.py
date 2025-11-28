from flask import Flask, request, jsonify, send_from_directory
import os
import uuid

app = Flask(__name__)

# Folder to save uploaded and processed videos
UPLOAD_FOLDER = "/tmp/uploads"
PROCESSED_FOLDER = "/tmp/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Dummy analysis function (replace with your actual cricket pose analysis)
def run_cricket_analysis(input_path):
    # For demo, just copy the file to processed folder with new name
    filename = os.path.basename(input_path)
    processed_name = f"processed_{uuid.uuid4().hex}_{filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
    # Here you would run your actual analysis code
    # For now, just copy the uploaded file
    import shutil
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

    # Generate URL for frontend (serve via /processed/<filename>)
    analysis_url = f"/processed/{processed_filename}"

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
