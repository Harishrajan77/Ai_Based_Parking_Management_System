# 🚗 AI-Based Parking Management System (ParkSight Intelligence)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)

## 📖 Project Overview
The **AI-Based Parking Management System (ParkSight Intelligence)** is an end-to-end computer vision solution designed to monitor and manage parking lot occupancy in real-time. By leveraging a state-of-the-art YOLOv8 object detection model, the system accurately classifies parking spaces as either "Empty" or "Occupied."

This project features a complete machine learning pipeline—from automated dataset preparation and GPU-accelerated model training to a modern web interface where users can upload images or video feeds for real-time analysis.

## ✨ Key Features
* **Real-Time Occupancy Detection:** Processes live video streams and static images frame-by-frame to identify empty vs. occupied parking slots instantly.
* **Interactive Web Dashboard:** A Flask-based web application with a responsive Tailwind CSS frontend for seamless media uploads and real-time MJPEG video streaming.
* **Predictive Forecasting:** Incorporates a Scikit-Learn Multiple Linear Regression model to forecast expected occupancy rates based on historical data, current capacity, and entry/exit flow.
* **Custom ML Pipeline:** Includes fully automated scripts to generate train/val/test splits, write YAML configurations, and train the YOLO model using PyTorch and GPU acceleration.
* **Professional HUD Integration:** Utilizes OpenCV to draw custom Heads-Up Displays (HUD) directly onto processed frames.

## 🛠️ Tech Stack & Tools Used
* **Languages:** Python (3.10), HTML5, CSS3 (Tailwind), JavaScript
* **Deep Learning & CV:** Ultralytics YOLOv8, PyTorch, OpenCV (`cv2`)
* **Data Science:** Scikit-Learn, NumPy
* **Backend Web Server:** Flask, Werkzeug

## 🗂️ Project Structure
```text
AI_Based_Parking_Management_System/
│
├── Video/
│   ├── model_pipeline/
│   │   ├── app.py                     # Main Flask web application
│   │   ├── prepare_yolo_dataset.py    # Auto-generates splits and data.yaml
│   │   ├── realtime_occupancy.py      # Desktop real-time OpenCV app with ML forecasting
│   │   └── test.py                    # Inference testing script
│   │
│   ├── train_parking_detector.py      # GPU-accelerated YOLOv8 training script
│   ├── requirements.txt               # Python dependencies
│   └── runs/                          # Contains compiled model weights (best.pt)
│
├── .gitignore                         # Configured to ignore raw datasets and large media
└── README.md                          # Project documentation
```

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Harishrajan77/Ai_Based_Parking_Management_System.git
cd AI_Based_Parking_Management_System/Video
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Web Application**
```bash
python model_pipeline/app.py
```
Navigate to `http://127.0.0.1:5000` in your web browser to access the dashboard.

**4. Run the Real-Time Desktop App**
To run the localized OpenCV window with predictive linear regression analytics:
```bash
python model_pipeline/realtime_occupancy.py --source your_video.mp4
```

## 🧠 Model Training
To retrain the model on new data:
1. Place your raw images in `Video/images/` and YOLO-formatted TXT labels in `Video/labels/`.
2. Run `python model_pipeline/prepare_yolo_dataset.py` to auto-generate the splits.
3. Run `python train_parking_detector.py` to begin GPU-accelerated training.

---
*Designed and developed as a robust AI portfolio project.*