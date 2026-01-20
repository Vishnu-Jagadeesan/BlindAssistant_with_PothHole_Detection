# BlindAssistant_with_PothHole_Detection
---

# 🛰️ Project "Vision" – Real-Time Pothole & Object Detection

![React](https://img.shields.io/badge/frontend-react-blue?logo=react)
![ONNX](https://img.shields.io/badge/model-YOLOv8%20(ONNX)-blueviolet)
![TensorFlow Lite](https://img.shields.io/badge/object%20detection-TFLite-orange?logo=tensorflow)
![License](https://img.shields.io/badge/license-%C2%A9%20Vishnu%20Jagadeesan-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

> A lightweight and smart vision system using ONNX + TFLite for browser-based pothole and object detection. Optimized for edge devices and accessibility.

---

## 📚 Table of Contents

- [🚀 Overview](#-project-vision--real-time-pothole--object-detection)
- [🔍 Features](#-features)
- [🧠 Model Details](#-model-details)
- [🗂️ Related Private Repositories](#️-related-private-repositories)
- [🛠️ Tech Stack](#️-tech-stack)
- [📸 Demo Preview](#-demo-preview)
- [📥 Request Access](#-request-access)
- [📄 License](#-license)

---

## 🔍 Features

- 🕳️ **Pothole Detection** using custom-trained YOLOv8 model in ONNX format  
- 👁️ **Object Detection** via TensorFlow Lite COCO-SSD for lightweight inference  
- 🎥 **Live Camera Integration** (browser-based with permission toggle)  
- 🔊 **Voice Alerts** for potholes or close-range obstacles  
- 🧠 **ONNX + TFLite Model Switching** for different vision tasks  
- 📱 **Optimized for Edge Devices & Mobile Web**  
- 🌐 **React.js Frontend with TailwindCSS and Framer Motion UI**

---

## 🧠 Model Details

### 📌 Pothole Detection (ONNX Model)
- Trained using **YOLOv8** in **PyTorch**
- Custom dataset with image segmentation masks and bounding boxes
- Exported to `.onnx` format for efficient and portable inference

### 📌 Object Detection (TFLite)
- Uses pre-trained **COCO-SSD** TensorFlow Lite model
- Great for detecting common objects (e.g., person, car, bike)
- Fast and reliable inference using `tflite` or `tfjs`

---

## 🗂️ Related Private Repositories

These repositories are part of the full training and deployment pipeline.  
📌 **Note**: All below repositories are **PRIVATE** – _request access if needed._

| Repository | Purpose | Access |
|-----------|---------|--------|
| 🔗 [PothHole_detection_ImgSegmentation](https://github.com/Vishnu-Jagadeesan/PothHole_detection_ImgSegmentation) | Code for dataset processing, annotation, YOLOv8 training | 🔒 Private |
| 🔗 [poth-hole_detection_using-trained-data](https://github.com/Vishnu-Jagadeesan/poth-hole_detection_using-trained-data) | ONNX model integration, inference scripts | 🔒 Private |
| 🔗 [Object-detection-with-tensorflowl-ite](https://github.com/Vishnu-Jagadeesan/Object-detection-with-tensorflowl-ite) | TFLite object detection with webcam preview | 🔒 Private |

📧 _Request access by opening an issue or contacting the author via GitHub/Gmail/LinkedIn/[Portfolio Website](https://vishnujagadeesan.com)._

---

## 🛠️ Tech Stack

| Layer        | Tools / Frameworks |
|--------------|--------------------|
| Frontend     | React.js (Vite), TailwindCSS, Framer Motion |
| Backend (Optional) | Flask (for ONNX runtime) |
| Vision Models | YOLOv8 (ONNX), TensorFlow Lite (COCO-SSD) |
| Accessibility | PyTesseract (OCR), Web Speech API |
| Deployment    | Render / GitHub Pages / Localhost |

---

## 📸 Demo Preview

> _Add visuals here if available:_

- Live camera view with bounding boxes
- Real-time voice alerts: “Pothole detected!” or “Person within 5m”
- Object detection text output in a dedicated panel

---

## 📥 Request Access

This repository references private training and deployment modules.  
To gain access:

1. Visit the linked repositories.
2. Click **"Request Access"** or open an issue in this repo.
3. Clearly state your purpose (academic, research, etc.)

🔗 [GitHub Profile – Vishnu Jagadeesan](https://github.com/Vishnu-Jagadeesan)

> You can also contract me :

- Sending an email to [vishnuj.cs.ug@gmail.com](mailto:vishnujagadeesan10@gmail.com)
- Connecting on [LinkedIn](https://www.linkedin.com/in/vishnu-jagadeesan/)
- Visiting the [Portfolio Website](https://vishnujagadeesan.com)

---

## 📄 License

This project is licensed under the [Vishnu Jagadeesan](LICENSE).
📄 License: Vishnu Jagadeesan| 🔒 All rights reserved


> © 2025 Vishnu Jagadeesan – For academic, research, and ethical use only.

---
