# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D20-IRL-detection is an edge AI/IoT system that physically rolls a D20 dice using a servo motor, captures the result via camera, and classifies the dice face (1–20) using a TensorFlow Lite model on a Raspberry Pi. It exposes results via a Flask web API and HTML UI.

## Setup

**Raspberry Pi (runtime):**
```bash
python3 -m venv venv && source venv/bin/activate
./tflite1/get_pi_requirements.sh   # installs OpenCV + system libs
pip install -r requirements-pi.txt
cp .env.example .env
```

**PC (training):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 main.py   # Flask app at http://localhost:5000
```

Camera calibration (Raspberry Pi with USB camera):
```bash
./fixcam.sh
```

Standalone detection scripts (for testing model without the full app):
```bash
python3 tflite1/TFLite_detection_image.py  --modeldir=tflite1/custom_model_lite --image=test.jpg
python3 tflite1/TFLite_detection_webcam.py --modeldir=tflite1/custom_model_lite
python3 tflite1/TFLite_detection_video.py  --modeldir=tflite1/custom_model_lite --video=test.mp4
```

## Architecture

Three core Python modules wire together the full pipeline:

| File | Role |
|------|------|
| `main.py` | Flask app — routes `GET /`, `POST /`, `POST /api/roll` |
| `dice.py` | Hardware logic — GPIO servo trigger, OpenCV capture, motion detection, GIF generation |
| `object_detection.py` | TFLite wrapper — loads `detect.tflite`, runs inference, draws bounding boxes |

**Data flow for a roll:**
1. `POST /api/roll` → `dice.roll_dice()` pulses GPIO pin 6 (servo)
2. OpenCV captures frames until motion stops
3. Final frame saved as JPEG; optionally 25 frames → GIF (debug mode)
4. `ObjectDetector.detect_objects()` runs TFLite inference on the frame
5. JSON response returned with image path, GIF path, detections, and timing

## Environment Variables

Defined in `.env` (see `.env.example`):
- `RESULT_FOLDER` — where output images/GIFs are saved (default: `results`)
- `MODEL_FOLDER` — path to the TFLite model directory (default: `tflite1/custom_model_lite`)

## Model

- `tflite1/custom_model_lite/detect.tflite` — active TFLite model (swap this file to update the model)
- Labels are hardcoded as `['1'..'20']` in `object_detection.py` — no labelmap file needed
- Architecture: **YOLOv8n** (replacing the original SSD-MobileNet v2 FPN-lite)
- Output format: `[1, 24, 8400]` → transposed to `[8400, 24]` (4 bbox + 20 class scores)

## Training Pipeline (PC with GPU)

```bash
pip install ultralytics

# 1. Convert Pascal VOC annotations to YOLO format
python scripts/convert_voc_to_yolo.py

# 2. Train YOLOv8n (~5 min on 3080)
yolo train model=yolov8n.pt data=data/data.yaml epochs=150 imgsz=640 batch=16 patience=30

# 3. Export to INT8 TFLite for Pi
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 int8=True

# 4. Deploy
cp runs/detect/train/weights/best_saved_model/best_int8.tflite tflite1/custom_model_lite/detect.tflite
```

Training data: `dice_training/` — 329 images with Pascal VOC XML annotations, ~570 instances across 20 classes.

## Hardware Dependencies

`RPi.GPIO` and `cv2` (OpenCV) are required at runtime on Raspberry Pi. The Flask app will fail gracefully if GPIO is unavailable (development on non-Pi hardware requires mocking or skipping hardware paths).
