import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_model(image_path):
    model = YOLO("models/best.onnx", task='detect')
    img = cv2.imread(image_path)
    results = model(img, conf=0.5)
    
    for res in results:
        print("\nDetection results:")
        print(f"Classes: {res.boxes.cls}")
        print(f"Confidences: {res.boxes.conf}")
        print(f"Coordinates: {res.boxes.xyxy}")
        
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf.item()
            print(f"Detected: {model.names[cls_id]} ({conf*100:.1f}%)")

if __name__ == "__main__":
    test_yolo_model("test_images/pothhole.jpeg")
    test_yolo_model("test_images/clear_road.jpeg")