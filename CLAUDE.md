# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D20-IRL-detection is an edge AI/IoT system that physically rolls D20 dice using a DC motor, captures the result via USB camera, and classifies the dice faces (1–20) using a YOLOv8n TFLite model on a Raspberry Pi. Results are exposed via a Flask web API and HTML UI.

## Setup

**Raspberry Pi (runtime):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-pi.txt
cp .env.example .env
# Set CAMERA_ROI via browser: python3 main.py → http://<pi-ip>:5000/calibrate
```

**PC (training):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 main.py   # Flask app at http://0.0.0.0:5000
```

## Architecture

| File | Role |
|------|------|
| `main.py` | Flask app — `GET /`, `POST /api/roll`, `GET /calibrate`, `GET /label` |
| `dice.py` | Hardware logic — GPIO motor trigger, OpenCV capture, motion detection, GIF generation |
| `object_detection.py` | TFLite wrapper — loads `detect.tflite`, runs YOLOv8 inference, draws bounding boxes |

**Data flow for a roll:**
1. `POST /api/roll` → `dice.roll_dice()` pulses GPIO pin 6 (motor)
2. Camera captures at 1920×1080; ROI crop applied; motion detection on 480×270 downscale
3. Final full-res ROI frame saved as JPEG when motion stops
4. `ObjectDetector.detect_objects()` runs TFLite inference
5. JSON response with image path, GIF path, detections, and timing

## Environment Variables

Defined in `.env` (see `.env.example`):
- `RESULT_FOLDER` — where output images/GIFs are saved under `static/` (default: `results`)
- `MODEL_FOLDER` — path to the TFLite model directory (default: `tflite1/custom_model_lite`)
- `CAMERA_ROI` — crop region `x,y,w,h` at 1920×1080 — set via `/calibrate` or `scripts/select_roi.py`

## Model

- `tflite1/custom_model_lite/detect.tflite` — active TFLite model (swap to update)
- Labels hardcoded as `['1'..'20']` in `object_detection.py` — no labelmap file needed
- Architecture: **YOLOv8n**
- Output: `[1, 24, 8400]` → transposed to `[8400, 24]` (4 bbox + 20 class scores)
- Bounding box coordinates are **normalized (0–1)** — scale by `imW`/`imH` directly
- Input dtype: `float32`, normalized to 0–1

## Training Pipeline (PC with GPU)

The model must be trained on your specific physical setup. Training on Pi is not supported.

```bash
# 1. Collect images (Pi)
python3 scripts/collect_training_data.py --count 100
# → saves to to_label/

# 2. Label images (browser at http://<pi-ip>:5000/label)
# → saves images + Pascal VOC XML to dice_training/

# 3. Convert annotations (PC)
python3 scripts/convert_voc_to_yolo.py
# → creates data/images/ and data/labels/

# 4. Train (~5 min on RTX 3080)
yolo train model=yolov8n.pt data=data/data.yaml epochs=150 imgsz=640 batch=16 patience=30

# 5. Export to INT8 TFLite
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 int8=True

# 6. Deploy
cp runs/detect/train/weights/best_saved_model/best_int8.tflite tflite1/custom_model_lite/detect.tflite
git add tflite1/custom_model_lite/detect.tflite && git commit -m "Update model" && git push
# On Pi: git pull
```

Training data: `dice_training/` — Pascal VOC XML annotations. Minimum ~100 images from your exact setup for reliable detection.

## Camera

- Capture resolution: 1920×1080 MJPG
- Motion detection runs on 480×270 downscale for Pi performance
- ROI crop configured via `CAMERA_ROI` env var (set in `/calibrate` UI)
- `fixcam.sh` locks white balance, sharpness, and power line frequency for consistent captures
- `scripts/select_roi.py` — CLI alternative to set ROI

## Hardware Dependencies

`RPi.GPIO` and `cv2` (OpenCV) are required at runtime on Raspberry Pi. The app will fail on import if GPIO is unavailable (non-Pi development requires mocking or skipping hardware paths).
