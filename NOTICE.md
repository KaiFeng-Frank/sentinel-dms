# NOTICE — Derivative Work Attribution

This repository is a **derivative work** based on:

> **tyrerodr/real-time-drowsy-driving-detection**
> https://github.com/tyrerodr/real-time-drowsy-driving-detection
> Author: Eng. Tyrone Eduardo Rodriguez Motato (tyrerodr@hotmail.com)

The upstream repository does not ship an explicit `LICENSE` file. This fork
is published publicly in good faith for **research, educational and
demonstration** purposes, with full attribution to the original author. No
commercial use is intended.

## Components inherited from upstream

- `DrowsinessDetector.py` (core YOLO+MediaPipe loop, heavily modified here)
- `AutoLabelling.py`, `CaptureData.py`, `LoadData.ipynb`,
  `RedirectData.ipynb`, `train.ipynb`
- YOLOv8 model weights under `runs/detecteye/` and `runs/detectyawn/`
- `requirements.txt`, `dataset.yaml`

## New components added by this fork

- `slow_system.py` — VLM Slow System (Qwen3.5-Omni via OpenAI-compatible API)
- `decision_fusion.py` — confidence-weighted Fast/Slow drowsiness fusion
- Substantial rewrite of `DrowsinessDetector.py`:
  - Multi-dimensional dark-theme product UI
  - MediaPipe-derived EAR + sliding-window PERCLOS
  - Slow System integration + frame submission + polling
- `NOTICE.md` (this file)
- Augmented `.gitignore`
- Extended `README.md` with Fast+Slow architecture documentation

## Takedown request

If the upstream author objects to this public fork, please open an issue on
this repository or contact the fork owner (Frank-tech1 on GitHub) and the
repository will be moved to private or removed promptly.
