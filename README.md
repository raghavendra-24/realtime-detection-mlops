# Real-Time Object Detection with MLOps Monitoring

A production-ready object detection system with drift monitoring, deployed as a containerized web service.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## 🎯 Features

- **Real-Time Detection**: YOLOv8 Nano trained on CrowdHuman (~117 FPS on GPU)
- **Drift Detection**: Automatic monitoring for input distribution shifts
- **MLOps Monitoring**: Prometheus metrics + Grafana dashboard
- **Web Interface**: Interactive Streamlit demo with webcam support
- **REST API**: FastAPI backend for production integration
- **Containerized**: Docker Compose for easy deployment

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 66.2% |
| Precision | 80.4% |
| Recall | 56.0% |
| Inference | 8.5ms (GPU) |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for live demo)

### Local Setup

```bash
cd realtime-detection-mlops
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Add model files to models/ folder:
# - crowdhuman_yolov8n_best.pt
# - baseline_stats.json

# Run
streamlit run frontend/app.py
```

### Docker

```bash
docker-compose up --build
# Frontend: http://localhost:8501
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Upload model to Hugging Face Hub:**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/crowdhuman-yolov8n ./models/
   ```

2. **Push code to GitHub**

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repo
   - Set main file: `frontend/app.py`
   - Add secret: `HF_MODEL_REPO = "your-username/crowdhuman-yolov8n"`

### Option 2: Railway/Render

1. Push to GitHub
2. Connect to Railway/Render
3. Set environment variables:
   - `MODEL_URL`: Direct URL to model file
   - `BASELINE_URL`: Direct URL to baseline_stats.json

## 📁 Project Structure

```
realtime-detection-mlops/
├── models/                    # Model files
├── backend/                   # FastAPI backend
│   ├── main.py               # API endpoints
│   ├── inference.py          # YOLOv8 engine
│   ├── drift_detector.py     # Drift detection
│   └── metrics.py            # Prometheus metrics
├── frontend/app.py           # Streamlit UI
├── monitoring/               # Prometheus + Grafana
├── docker-compose.yml
└── requirements.txt
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Run detection on image |
| `/metrics` | GET | Prometheus metrics |

## 📝 Resume Highlights

> **Real-Time Object Detection System with MLOps Monitoring**
> - Trained YOLOv8 on CrowdHuman achieving **66.2% mAP@50** with **117 FPS**
> - Implemented **drift detection** using statistical analysis
> - Built REST API with **FastAPI** and demo with **Streamlit**
> - Integrated **Prometheus/Grafana** monitoring with alerts
> - Containerized with **Docker Compose**

## 📄 License

MIT License
