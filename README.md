# Drowsiness Detection System — Fast + Slow VLM Edition

> **This is a derivative work** forked from
> [tyrerodr/real-time-drowsy-driving-detection](https://github.com/tyrerodr/real-time-drowsy-driving-detection).
> Upstream repo: YOLOv8 + PyQt5 real-time drowsiness detector.
> This fork adds a **Slow System** layer powered by a vision-language model
> (Qwen3.5-Omni via DashScope), plus a multi-dimensional decision fusion
> module and a product-grade dark-theme UI.

## Fast + Slow architecture

| Layer | Runs at | Technology | What it does |
|---|---|---|---|
| **Fast System** | ~30 FPS | MediaPipe + YOLOv8 | Real-time PERCLOS, EAR, blinks, microsleeps, yawn counting |
| **Slow System** | every 10 s | Qwen3.5-Omni (VLM) | Multi-dimensional reasoning: drowsiness (body posture, facial muscle tone, head tilt), distraction (phone / eating / looking away), anomaly (drug/alcohol signs, emotional distress), occlusion self-awareness, scene context (lighting, passengers), recommended action |
| **Decision Fusion** | every frame | `decision_fusion.py` | Confidence-weighted fusion of drowsiness only. Other VLM dimensions pass through directly (Fast System has no equivalent signal) |

## Configuring the Slow System (VLM)

The Slow System calls an OpenAI-compatible endpoint. Credentials are read
from environment variables — **no keys are hardcoded in the source**:

```bash
export DASHSCOPE_API_KEY=sk-your-key-here
# optional
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_MODEL=qwen3.5-omni-plus
```

If `DASHSCOPE_API_KEY` is unset the Slow System automatically falls back to
**mock mode** so you can exercise the full pipeline offline. The repo ships
with a complete mock implementation that returns plausible multi-dimensional
responses.

## Files added by this fork

- `slow_system.py` — background VLM worker (thread, non-blocking, mock + real path)
- `decision_fusion.py` — confidence-weighted drowsiness fusion
- `DrowsinessDetector.py` — rewritten with multi-dim dark-theme UI
- `.gitignore` — augmented with Python / secret hygiene rules

---

![image](https://github.com/user-attachments/assets/81ab2ce9-94ed-479b-bb76-d289c99800fc)
![image](https://github.com/user-attachments/assets/0615e219-f623-47ff-9448-946a9c273500)
![image](https://github.com/user-attachments/assets/b25705ed-d976-45a3-a080-fe1e12f220fd)

## Overview

The **Drowsiness Detection System** is a project designed to monitor a person's alertness in real-time by analyzing facial features.  
By utilizing computer vision and machine learning techniques, the system aims to detect signs of drowsiness and provide timely alerts — particularly useful for applications like driver monitoring.

This repository focuses on illustrating the full development process, including data capture, auto-labeling, model training, and detection pipeline integration.

---

## Features

- **Real-time Monitoring**: Detects signs of drowsiness using a webcam or video input.
- **Dual Model Detection**: Separate YOLOv8 models for eye closure detection and yawning detection.
- **Facial Landmarks Analysis**: Tracks eye status, head position, and mouth movements.
- **Data Capture Pipeline**: Tools to collect and organize custom datasets.
- **Auto Labeling with GroundingDINO**: Automated bounding box generation.
- **User Interface**: Built with PyQt5 for real-time visualization and alerts.

---

## Key Files

- `AutoLabelling.py`: Script for automated bounding box labeling using GroundingDINO.
- `CaptureData.py`: Records and logs video data for analysis or training.
- `DrowsinessDetector.py`: Core detection script integrating real-time inference and alerts.
- `LoadData.ipynb`: Loads and preprocesses datasets.
- `RedirectData.ipynb`: Organizes and redirects captured data for training.
- `train.ipynb`: Notebook for training the YOLO models.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tyrerodr/Real_time_drowsy_driving_detection.git
    cd Real_time_drowsy_driving_detection
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the detection system:**
    ```bash
    python DrowsinessDetector.py
    ```

---

## Usage

- **Real-Time Detection:** Run `DrowsinessDetector.py` with a connected webcam to monitor drowsiness.
- **Data Capture:** Use `CaptureData.py` to collect video frames for training or testing.
- **Training New Models:** Use `train.ipynb` to retrain the models on your custom datasets.

---

## How It Works

The system uses two separate YOLOv8 models:

1. **Eye Detection Model:**
   - Classifies eyes as open or closed.
   - Trained on public datasets:
     - [Eyes Dataset](https://www.kaggle.com/datasets/charunisa/eyes-dataset/code)
     - [MRL Eye Dataset](https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset)
   - ~53,000 images for training, ~3,000 images for validation.

2. **Yawning Detection Model:**
   - Detects yawning (mouth open) vs not yawning (mouth closed).
   - Trained on:
     - [Yawning Dataset](https://www.kaggle.com/datasets/deepankarvarma/yawning-dataset-classification?select=yawn)

**Auto Labeling:**  
GroundingDINO was used to generate bounding boxes for YOLO training to improve dataset quality.

Once trained, the models' predictions are combined with confidence thresholds and visualized in a PyQt5 GUI.

---

## Technologies Used

- **Python**
- **YOLOv8** – Object detection framework.
- **OpenCV** – Computer vision tasks.
- **GroundingDINO** – Auto-labeling tool.
- **TensorFlow / Keras** – Model training.
- **PyQt5** – Graphical user interface.

---

## Important Note

This repository is intended primarily to showcase the development process of a drowsiness detection system — including data collection, model training, and real-time integration.

The uploaded model weights are **preliminary** and **not fully trained to convergence**.  
They are mainly for demonstration purposes, and final production-ready models are maintained separately.

We appreciate any feedback and contributions to improve the system.

---

## Future Improvements

- **Integration with Wearables:** Add heart rate or other vitals monitoring.
- **Multi-Person Detection:** Extend detection to multiple subjects simultaneously.
- **Mobile Deployment:** Create a mobile app version for real-time on-the-go monitoring.

---

**Eng. Tyrone Eduardo Rodriguez Motato**  
Computer Vision Engineer  
Guayaquil, Ecuador  
Email: tyrerodr@hotmail.com
