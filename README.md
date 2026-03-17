# Real-time D20 Dice Detection with Raspberry Pi

Physical D20 dice roller with real-time face detection. A servo motor rolls the dice, a USB camera captures the result, and a YOLOv8n TFLite model running on a Raspberry Pi classifies which face is showing (1–20). Results are exposed via a Flask REST API and web UI.

![](README/img.png)
![](3d_model/imgs/hardware.gif)
![](README/output.gif)
![](README/output.jpg)

## Hardware

- Raspberry Pi 3B+ or 4
- USB webcam (Logitech C920 or similar with manual focus support recommended)
- DC motor connected to GPIO pin 6 (via relay)
- 3D printed dice cup (STL files in `3d_model/stl/`)

## Setup

### Raspberry Pi (runtime)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-pi.txt
cp .env.example .env
```

Configure the camera crop region via the browser UI:
```bash
python3 main.py
# then open http://<pi-ip>:5000/calibrate
```

Or using the CLI script:
```bash
python3 scripts/select_roi.py
# paste the printed CAMERA_ROI value into .env
```

Optionally fix camera settings (exposure, white balance):
```bash
./fixcam.sh
```

### PC (training)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 main.py   # Flask app at http://<pi-ip>:5000
```

## Web UI Routes

| Route | Description |
|-------|-------------|
| `GET /` | Roll page — trigger a roll and see the result |
| `GET /calibrate` | Live camera preview — set and save the ROI crop |
| `GET /label` | Label training images collected from the Pi |

## API

### POST /api/roll

Triggers the motor, captures a frame when motion stops, runs detection.

```json
{
  "detections": "7 and 14",
  "image": "static/results/<uuid>.jpg",
  "gif": "static/results/<uuid>.gif",
  "time_elapsed": 2.37,
  "time_elapsed_detection": 0.25
}
```

Add `debug=true` in the form body to generate a GIF of the roll.

## Environment Variables

See `.env.example`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RESULT_FOLDER` | `results` | Where output images/GIFs are saved (under `static/`) |
| `MODEL_FOLDER` | `tflite1/custom_model_lite` | Path to the TFLite model directory |
| `CAMERA_ROI` | _(empty)_ | Crop region `x,y,w,h` — set via `/calibrate` or `select_roi.py` |

## Model

**YOLOv8n** exported to INT8 TFLite for edge inference on Raspberry Pi.

- Model file: `tflite1/custom_model_lite/detect.tflite`
- Input: `640×640` RGB, float32 normalized (0–1)
- Output: `[1, 24, 8400]` — transposed to `[8400, 24]` (4 bbox coords + 20 class scores)
- Labels: hardcoded as `['1'..'20']` — no labelmap file needed
- ~3.2 MB, ~50ms inference on Pi 4

## Training Pipeline

The model must be trained on **your specific setup** (camera angle, dice type, lighting, bowl). The pipeline runs on a PC with a GPU.

### Step 1 — Collect training images (Pi)

Auto-rolls the dice N times and saves each captured frame to `to_label/`:

```bash
python3 scripts/collect_training_data.py --count 100
```

### Step 2 — Label images (browser)

Open `http://<pi-ip>:5000/label` while `main.py` is running.

For each image:
1. **Click and drag** to draw a bounding box around each die
2. **Click the face number** (1–20) for each box
3. **Save & Next** — saves image + Pascal VOC XML to `dice_training/`
4. **Skip** — discards bad images (blurry, dice on edge, etc.)

> Two dice per image is recommended — it matches the final usage and doubles labeling efficiency.

### Step 3 — Convert annotations (PC)

Converts Pascal VOC XML → YOLO format and splits into train/val sets:

```bash
python3 scripts/convert_voc_to_yolo.py
```

### Step 4 — Train (PC with GPU)

```bash
yolo train model=yolov8n.pt data=data/data.yaml epochs=150 imgsz=640 batch=16 patience=30
```

~5 minutes on an RTX 3080. Training results saved to `runs/detect/train/`.

### Step 5 — Export to TFLite INT8

```bash
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 int8=True
```

### Step 6 — Deploy to Pi

```bash
cp runs/detect/train/weights/best_saved_model/best_int8.tflite tflite1/custom_model_lite/detect.tflite
git add tflite1/custom_model_lite/detect.tflite
git commit -m "Update model"
git push
```

On Pi:
```bash
git pull
```

### Retraining tips

- **Minimum ~100 labeled images** from your exact setup for reliable detection
- **Two dice per image** preferred — more annotations per roll
- After changing camera position or distance, collect new data and retrain from scratch
- Run `scripts/select_roi.py` or use `/calibrate` after any camera repositioning

## 3D Models

STL files for the dice cup and mechanical parts are in `3d_model/stl/`. A revised version with sloped inner walls (to guide dice toward center) is planned.

## TODO

- [ ] Revised 3D model with sloped bowl walls so dice settle in center
- [ ] Improve motion detection robustness (background subtraction)
- [ ] API authentication for external usage
- [ ] Queue system for concurrent requests
