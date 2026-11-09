# 🚧 AI-Powered Road Damage Detection Using YOLO11

An intelligent computer vision application that detects potholes in road images and videos using YOLO11n. The project combines deep learning and Streamlit to provide an interactive road damage detection system.

## 📌 Project Overview

Road damage, particularly potholes, creates safety risks for drivers and pedestrians. Traditional road inspection methods require manual monitoring and can be time-consuming.

This project uses YOLO11n (You Only Look Once) to automatically detect potholes from road images and videos.

The trained model is integrated into a Streamlit web application where users can upload road images, visualize detected potholes, view confidence scores, and download annotated results.

## 🎯 Objectives

- Automate pothole detection using computer vision.
- Detect road damage from images and videos.
- Evaluate object detection performance using standard metrics.
- Build an interactive web application.
- Explore practical applications of deep learning in road infrastructure monitoring.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLO11n | Object detection |
| PyTorch | Deep learning framework |
| Ultralytics | YOLO implementation |
| OpenCV | Image and video processing |
| Streamlit | Web application |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |

## 🧠 Model Information

- Architecture: YOLO11n
- Task: Object Detection
- Input Resolution: 800 × 800
- Training Epochs: 100
- Hardware: NVIDIA Tesla T4 GPU
- Number of Parameters: Approximately 2.6 million

## 📊 Model Performance

The following results were obtained during model evaluation:

| Metric | Score |
|---|---:|
| Precision | 89.5% |
| Recall | 83.9% |
| mAP@50 | 91.0% |
| mAP@50-95 | 57.9% |

These metrics represent the reported validation results for the 800-resolution experiment.

## ⚙️ Project Workflow

1. Dataset collection and preparation.
2. Exploratory Data Analysis (EDA).
3. Annotation validation and cleaning.
4. YOLO11n model training.
5. Model evaluation.
6. Confidence threshold experimentation.
7. Image and video inference.
8. Streamlit application development.

## 📁 Project Structure

```text
Road-Damage-Detection/
│
├── App.py
├── best.pt
├── requirements.txt
├── Notebook.ipynb
├── README.md
├── .gitignore
│
└── assets/
    ├── prediction.png
    └── processed_video.mp4
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/road-damage-detection.git
```

### 2. Navigate to the project

```bash
cd road-damage-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run App.py
```

The application will open at:

```text
http://localhost:8501
```

## 💻 Application Features

- Upload road images.
- Detect multiple potholes.
- Visualize bounding boxes.
- Adjust confidence thresholds.
- Configure NMS IoU thresholds.
- View total detections.
- View average confidence.
- Download annotated images.
- Upload and process road videos.

## 🔬 Experiments

| Experiment | Architecture | Image Size | mAP@50 |
|---|---|---:|---:|
| E1 | YOLO11n | 640 | 88.8% |
| E2 | YOLO11n | 800 | 91.0% |

Higher input resolution improved the reported detection metrics in the second experiment.

## 🔮 Future Improvements

- Improve detection of small potholes.
- Reduce false-positive detections.
- Explore object tracking across video frames.
- Integrate GPS-based road damage localization.
- Develop a road condition monitoring dashboard.
- Explore deployment on edge devices.

## 🙏 Acknowledgements

Special thanks to my mentor and trainer for their continuous guidance, support, and encouragement throughout this project.

Their practical insights helped me understand computer vision concepts and apply them to a real-world problem.

## 👨‍💻 Author

**Venu Madhav**

Computer Vision | Deep Learning | Machine Learning

---

⭐ If you find this project interesting, consider giving the repository a star!