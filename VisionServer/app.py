from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.lite.python.interpreter import Interpreter
import warnings
import logging
import os

os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

# Configuration
MODEL_PATHS = {
    'yolo': 'models/best.onnx',
    'coco': 'models/coco_ssd_mobilenet_v1_1.0_quant.tflite'
}
CLASS_FILTERS = {
    'coco': [1, 3, 8],  # person, car, truck (COCO IDs)
    'yolo': ['pothole']
}
DISTANCE_CONFIG = {
    'focal_length': 800,
    'known_widths': {'person': 50, 'car': 180, 'truck': 300, 'pothole': 30},
    'thresholds': {'person': 3.0, 'car': 5.0, 'truck': 7.0, 'pothole': 1.0}
}
CONFIDENCE_THRESHOLDS = {
    'yolo': 0.65,
    'coco': 0.35
}

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Global models
yolo_model = None
coco_interpreter = None
coco_labels = {}
coco_input_details = []
coco_output_details = []

def initialize_models():
    global yolo_model, coco_interpreter, coco_labels, coco_input_details, coco_output_details

    try:
        # Load YOLO
        yolo_model = YOLO(MODEL_PATHS['yolo'], task='detect')
        logging.info(f"YOLO classes: {yolo_model.names}")

        # Load COCO
        coco_interpreter = Interpreter(MODEL_PATHS['coco'])
        coco_interpreter.allocate_tensors()

        # Get model details
        coco_input_details = coco_interpreter.get_input_details()
        coco_output_details = coco_interpreter.get_output_details()
        logging.info(f"COCO input: {coco_input_details[0]['shape']}")
        logging.info(f"COCO outputs: {[d['name'] for d in coco_output_details]}")

        # Load labels
        with open('models/coco_labels.txt', 'r') as f:
            coco_labels = {idx+1: line.strip() for idx, line in enumerate(f) if line.strip()}

        logging.info(f"COCO labels loaded. Sample: 1={coco_labels[1]}, 3={coco_labels[3]}, 8={coco_labels[8]}")

    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        raise

initialize_models()

def calculate_distance(pixel_width, obj_type):
    if pixel_width <= 0:
        return float('inf')
    known_width = DISTANCE_CONFIG['known_widths'][obj_type]
    return (known_width * DISTANCE_CONFIG['focal_length']) / (pixel_width * 100)

def process_coco_detections(frame):
    alerts = []
    try:
        h, w = frame.shape[:2]

        # Prepare input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (300, 300))
        input_data = np.expand_dims(resized, 0).astype(np.uint8)

        # Inference
        coco_interpreter.set_tensor(coco_input_details[0]['index'], input_data)
        coco_interpreter.invoke()

        # Get outputs
        boxes = coco_interpreter.get_tensor(coco_output_details[0]['index'])[0]
        classes = coco_interpreter.get_tensor(coco_output_details[1]['index'])[0].astype(int)
        scores = coco_interpreter.get_tensor(coco_output_details[2]['index'])[0]
        num_det = int(coco_interpreter.get_tensor(coco_output_details[3]['index'])[0])

        for i in range(num_det):
            if scores[i] < CONFIDENCE_THRESHOLDS['coco']:
                continue

            class_id = classes[i] + 1  # Fix 0-based to 1-based

            if class_id not in CLASS_FILTERS['coco']:
                continue

            # Convert boxes [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            pw = xmax - xmin
            if pw < 10:
                continue

            label = coco_labels[class_id]
            distance = calculate_distance(pw, label)

            if distance < DISTANCE_CONFIG['thresholds'][label]:
                alerts.append({
                    'label': label,
                    'distance': distance,
                    'confidence': scores[i]
                })
                logging.info(f"COCO: {label} ({distance:.1f}m, {scores[i]:.2f})")

    except Exception as e:
        logging.error(f"COCO error: {str(e)}")

    return alerts

def process_yolo_detections(frame):
    alerts = []
    try:
        results = yolo_model(frame,
                             conf=CONFIDENCE_THRESHOLDS['yolo'],
                             iou=0.45)

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id].lower().strip()

                if label != 'pothole':
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pw = x2 - x1
                dist = calculate_distance(pw, label)

                if dist < DISTANCE_CONFIG['thresholds'][label]:
                    alerts.append({
                        'label': label,
                        'distance': dist,
                        'confidence': box.conf.item()
                    })
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
        # Read frame
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect objects
        coco_alerts = process_coco_detections(frame)
        yolo_alerts = process_yolo_detections(frame)

        # Combine results
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

    # Draw detections
    _ = process_coco_detections(frame.copy())
    _ = process_yolo_detections(frame.copy())

    cv2.imwrite('debug.jpg', frame)
    return send_file('debug.jpg', mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
