from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import warnings
import logging
import os

# Runtime setup for Ultralytics
os.makedirs('/tmp/ultralytics', exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.makedirs('/tmp/matplotlib', exist_ok=True)
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
MODEL_PATHS = {
    'yolo': 'models/best_final_quant_uint8.onnx',
    'coco': 'models/coco_ssd_mobilenet_v1_1.0_quant.tflite'
}
CLASS_FILTERS = {
    'coco': [1, 3, 8],
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Lazy model loaders
yolo_model = None
interpreter = None
labels = {}
input_details = []
output_details = []

def load_yolo():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATHS['yolo'], task='detect')
        logging.info("YOLO model loaded")

def load_coco():
    global interpreter, labels, input_details, output_details
    if interpreter is None:
        interpreter = Interpreter(MODEL_PATHS['coco'])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        with open('models/coco_labels.txt', 'r') as f:
            labels = {idx + 1: line.strip() for idx, line in enumerate(f) if line.strip()}
        logging.info("COCO model loaded")

def calculate_distance(pixel_width, obj_type):
    if pixel_width <= 0:
        return float('inf')
    known_width = DISTANCE_CONFIG['known_widths'][obj_type]
    return (known_width * DISTANCE_CONFIG['focal_length']) / (pixel_width * 100)

def process_coco_detections(frame):
    load_coco()
    alerts = []
    try:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (300, 300))
        input_data = np.expand_dims(resized, 0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_det = int(interpreter.get_tensor(output_details[3]['index'])[0])

        for i in range(num_det):
            if scores[i] < CONFIDENCE_THRESHOLDS['coco']:
                continue
            class_id = classes[i] + 1
            if class_id not in CLASS_FILTERS['coco']:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            pw = xmax - xmin
            if pw < 10:
                continue
            label = labels[class_id]
            distance = calculate_distance(pw, label)
            if distance < DISTANCE_CONFIG['thresholds'][label]:
                alerts.append({'label': label, 'distance': distance, 'confidence': scores[i]})
                logging.info(f"COCO: {label} ({distance:.1f}m, {scores[i]:.2f})")
    except Exception as e:
        logging.error(f"COCO error: {str(e)}")
    return alerts

def process_yolo_detections(frame):
    load_yolo()
    alerts = []
    try:
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLDS['yolo'], iou=0.45)
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
        coco_alerts = process_coco_detections(frame)
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
