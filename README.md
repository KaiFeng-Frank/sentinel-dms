# SENTINEL DMS — Fast + Slow Driver Monitoring

> **Derivative work** of
> [tyrerodr/real-time-drowsy-driving-detection](https://github.com/tyrerodr/real-time-drowsy-driving-detection).
> See [`NOTICE.md`](NOTICE.md) for full attribution and takedown policy.

A two-layer **Fast + Slow** driver monitoring system. The **Fast System**
(MediaPipe + YOLOv8) runs on every frame at ~30 FPS and tracks low-level
eye/mouth physiology (PERCLOS, EAR, blinks, microsleeps, yawns). The
**Slow System** is a vision-language model (Qwen3.5-Omni) that runs
continuously in a background thread and contributes high-level semantic
reasoning that the Fast System cannot do — distraction, anomaly detection,
occlusion self-awareness, and scene context — producing a fused
drowsiness score and a recommended action.

The whole thing is wrapped in **SENTINEL DMS**, a product-grade dark-theme
cockpit UI built with custom `QPainter` widgets (radial risk gauge with
animated needle, tactical HUD video, pill status chips, sparklines, etc).

---

## Two-layer architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │                 SENTINEL DMS — Driver Monitoring              │
   │                                                               │
   │   ┌──────────────┐       ┌──────────────────────────────┐   │
   │   │  Fast System │──────▶│     Decision Fusion          │   │
   │   │  30 FPS      │       │   (drowsiness dimension only)│──▶│
   │   │  MP + YOLOv8 │       │                              │   │
   │   │  PERCLOS,EAR │       │   fast_conf ≥ 0.8  → 0.8/0.2│   │
   │   └──────────────┘       │   fast_conf ≤ 0.5  → 0.2/0.8│   │
   │          │               │   in between   → linear    │   │
   │          │               └──────────────────────────────┘   │
   │          │                                                   │
   │          ▼ frame submit (10 Hz)                               │
   │   ┌──────────────┐                                           │
   │   │  Slow System │  ─▶  VLM  ─▶  {drowsiness, distraction,  │
   │   │  back-to-back│      Qwen    anomaly, occlusion, context,│
   │   │  ~0.3 Hz     │      3.5-omni overall_risk, explanation, │
   │   │              │      flash    recommended_action}         │
   │   └──────────────┘                                           │
   │                         distraction / anomaly / occlusion /  │
   │                         context bypass fusion — shown direct │
   └──────────────────────────────────────────────────────────────┘
```

| Layer | Rate | Tech | Outputs |
|---|---|---|---|
| **Fast System** | ~30 FPS | MediaPipe FaceMesh + YOLOv8 (eye / yawn) | PERCLOS, EAR, blinks, microsleeps, yawn count, fast-drowsiness 0-10 + confidence |
| **Slow System** | ~0.3-0.5 Hz continuous | Qwen3.5-Omni via DashScope OpenAI-compat | 5-dimension analysis + overall risk + recommended action |
| **Decision Fusion** | every frame | pure Python | Weighted drowsiness only; other VLM dims pass through |

## What the Slow System buys you

The Fast System is excellent at *"are the eyes closed right now?"* but it
can't answer *"is the driver looking at their phone? Are they intoxicated?
Is the mask occluding my judgment?"*. The Slow System VLM is prompted to
fill this exact schema:

```json
{
  "drowsiness":  { "level": 0..10, "confidence": 0..1 },
  "distraction": { "detected": bool, "type": "phone|eating|talking|looking_away|operating|other|none",
                   "confidence": 0..1 },
  "anomaly":     { "detected": bool, "description": "...|null", "severity": "none|low|medium|high" },
  "occlusion":   { "type": ["mask","sunglasses","hat","none"], "impact_on_reliability": 0..1 },
  "context":     { "lighting": "good|dim|dark", "passengers_detected": bool },
  "overall_risk": 0..10,
  "explanation": "natural-language report",
  "recommended_action": "none|verbal_warning|alarm|pull_over"
}
```

Only the `drowsiness` dimension is fused with the Fast System. The rest —
distraction, anomaly, occlusion, context, overall_risk, recommended_action
— are slow-only capabilities and are displayed directly in the UI.

## UI — SENTINEL DMS cockpit

The GUI is a single-window dark-theme HUD built with custom `QPainter`
widgets (zero external graphics libraries):

- **Brand header bar** — wordmark, LIVE / FAST-FPS / VLM-age status chips (LIVE blinks at 700 ms), live system clock
- **HUDVideoLabel** — 880 × 660 webcam feed with tactical cyan corner markers and center crosshair
- **RiskGauge** — radial arc gauge (270° sweep) with tick ring, eased value animation at 60 Hz, 72 px Arial Black central number, and a gradient recommended-action banner underneath
- **Fast System card** — 4 × 2 metric grid (blinks, microsleeps, yawns, PERCLOS, EAR, yawn-duration, fast-drowsiness, fast-confidence) with a rolling **Sparkline** of recent PERCLOS
- **Slow System card** — 5-row multi-dimensional analysis with unicode progress bars (`█`-block) for per-dimension confidence / severity / occlusion-impact
- **VLM analysis report** — full-width bottom strip showing the Qwen natural-language explanation
- **Status footer** — session timer, current model, FPS, GPU

Threading discipline: the capture / process worker threads only write
shared state under `_frame_lock`. A main-thread `QTimer` (`_ui_tick`) at
~30 Hz drives all widget updates — Qt thread-safe.

## Configuration (env vars only — no hardcoded keys)

```bash
# Required for real Qwen calls. If unset, SlowSystem falls back to mock mode
# automatically so the full UI pipeline still runs.
export DASHSCOPE_API_KEY=sk-your-key-here

# Optional — defaults shown
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_MODEL=qwen3.5-omni-flash   # flash is 3-4× faster than plus
export SLOW_INTERVAL_SECONDS=0              # 0 = back-to-back (max throughput)
export SLOW_IMAGE_MAX_SIDE=480              # JPEG long-side clamp
```

**Choosing a model** — measured DashScope latencies (480 px JPEG q80, 3 back-to-back cycles each):

| Model | First call | Warm | Effective rate |
|---|---|---|---|
| `qwen3.5-omni-flash` ← **default** | 4.7 s | **2.4 s** | ~3-4 calls / 10 s |
| `qwen3-omni-flash` | 4.8 s | 3.5 s | ~2 calls / 10 s |
| `qwen3.5-omni-plus` | 8.6 s | 10.7 s | ~1 call / 10 s |

Flash gives you maximum reaction speed at slightly lower reasoning depth;
`plus` gives you more thorough explanations at a quarter of the throughput.
Switch any time via `DASHSCOPE_MODEL`.

> ⚠️  **Cost & rate limits.** In max-throughput mode (`SLOW_INTERVAL_SECONDS=0`)
> the system fires ~1200 VLM calls per hour. DashScope bills per token.
> If you hit a 429 rate limit, the VLM chip in the header turns red
> (`VLM ERROR`) and the pipeline keeps going on Fast-only. Set
> `SLOW_INTERVAL_SECONDS=5` (or higher) to throttle.

## Quick start

```bash
# 1. Clone
git clone https://github.com/Frank-tech1/drowsy-driving-vlm.git
cd drowsy-driving-vlm

# 2. Environment (this fork was developed on Python 3.10 + CUDA 12.8)
conda create -n drowsy-det python=3.10 -y
conda activate drowsy-det
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Linux only: swap opencv-python for headless to avoid PyQt5 plugin clash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless opencv-contrib-python-headless
pip install openai   # for the Slow System

# 3. Run — with real Qwen
export DASHSCOPE_API_KEY=sk-your-key-here
python DrowsinessDetector.py

# Run — mock mode (no API key needed, full UI still exercised)
unset DASHSCOPE_API_KEY
python DrowsinessDetector.py
```

## Files added by this fork

| File | Purpose |
|---|---|
| `slow_system.py` | Background VLM worker thread. Mock + real OpenAI-compatible path. Back-to-back loop when `interval_seconds ≤ 0`. Image pre-scale before upload |
| `decision_fusion.py` | Confidence-weighted fusion of the drowsiness dimension only |
| `DrowsinessDetector.py` | Completely rewritten: custom Qt widgets (`RiskGauge`, `StatusChip`, `HUDVideoLabel`, `Sparkline`, `SectionCard`), thread-safe `_ui_tick` QTimer, multi-dimensional rendering, MediaPipe EAR + sliding-window PERCLOS, winsound/paplay cross-platform alert |
| `NOTICE.md` | Derivative-work attribution and takedown policy |
| `.gitignore` | Extended with Python bytecode + secret hygiene rules |

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
