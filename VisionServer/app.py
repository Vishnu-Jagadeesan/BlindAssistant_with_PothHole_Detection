from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import warnings
import logging
import os

# Environment setup
os.makedirs('/tmp/ultralytics', exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
MODEL_PATHS = {
    'yolo': 'models/best.onnx'
}
CLASS_FILTERS = {
    'yolo': ['pothole']
}
DISTANCE_CONFIG = {
    'focal_length': 800,
    'known_widths': {'pothole': 30},
    'thresholds': {'pothole': 1.0}
}
CONFIDENCE_THRESHOLDS = {
    'yolo': 0.65
}

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

def calculate_distance(pixel_width, obj_type):
    if pixel_width <= 0:
        return float('inf')
    known_width = DISTANCE_CONFIG['known_widths'][obj_type]
    return (known_width * DISTANCE_CONFIG['focal_length']) / (pixel_width * 100)

def process_yolo_detections(frame):
    alerts = []
    try:
        model = YOLO(MODEL_PATHS['yolo'])
        results = model(frame, conf=CONFIDENCE_THRESHOLDS['yolo'], iou=0.45)
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower().strip()
                if label != 'pothole':
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pw = x2 - x1
                dist = calculate_distance(pw, label)
                if dist < DISTANCE_CONFIG['thresholds'][label]:
                    alerts.append({'label': label, 'distance': dist, 'confidence': box.conf.item()})
                    logging.info(f"YOLO: {label} ({dist:.1f}m, {box.conf.item():.2f})")
    except Exception as e:
        logging.error(f"YOLO error: {str(e)}")
    return alerts

@app.route('/')
def index():
    return "Vision Backend is running", 200

@app.route('/process_frame', methods=['POST'])
def handle_frame():
    if 'frame' not in request.files:
        return jsonify({'alert': False, 'message': ''})
    try:
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        coco_alerts = []  # Disabled to save memory
        yolo_alerts = process_yolo_detections(frame)
        all_alerts = sorted(coco_alerts + yolo_alerts, key=lambda x: x['distance'])
        if all_alerts:
            closest = all_alerts[0]
            msg = f"Warning: {closest['label']} ahead ({closest['distance']:.1f}m)"
            return jsonify({'alert': True, 'message': msg})
        return jsonify({'alert': False, 'message': ''})
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return jsonify({'alert': False, 'message': ''})

@app.route('/debug', methods=['POST'])
def debug_detections():
    file = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    _ = process_yolo_detections(frame.copy())
    cv2.imwrite('debug.jpg', frame)
    return send_file('debug.jpg', mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
