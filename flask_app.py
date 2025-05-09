from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import threading
import uuid
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tracking

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'videos'
OUTPUT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Dictionary to keep track of processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(input_path, output_path, filename_key):
    try:
        logger.info(f"Starting processing for: {input_path}")
        tracking.main(input_path, output_path)
        logger.info(f"Processing completed for: {output_path}")
        processing_status[filename_key] = 'completed'
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        processing_status[filename_key] = 'error'

@app.route('/upload', methods=['POST'])
def upload_video():
    logger.info("Upload endpoint hit")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'Fayl tanlanmadi'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'Fayl tanlanmadi'}), 400

    if not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Noto\'g\'ri fayl formati'}), 400

    try:
        # Create unique filename
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        logger.info(f"Saving file to: {input_path}")
        file.save(input_path)
        logger.info("File saved successfully")
        
        # Mark status as processing
        processing_status[output_filename] = 'processing'
        
        # Start background thread to process video
        thread = threading.Thread(target=process_video, args=(input_path, output_path, output_filename))
        thread.start()
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'status_url': f'/status/{output_filename}'
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Server xatosi',
            'details': str(e)
        }), 500

@app.route('/status/<filename>')
def check_status(filename):
    status = processing_status.get(filename, 'processing')
    return jsonify({'status': status})

@app.route('/results/<filename>')
def serve_video(filename):
    try:
        logger.info(f"Serving video: {filename}")
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except FileNotFoundError:
        logger.error(f"Video not found: {filename}")
        return jsonify({'error': 'Video topilmadi'}), 404

@app.route('/results_page/<filename>')
def results_page(filename):
    return render_template('results.html', video_filename=filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
