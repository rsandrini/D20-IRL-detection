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
# Edit .env: set SECRET_KEY, optionally ADMIN_PASSWORD
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
| `main.py` | Flask app — all routes, `RollQueue` class, permission helpers |
| `dice.py` | Hardware logic — GPIO motor trigger, persistent camera, motion detection, GIF generation |
| `object_detection.py` | TFLite wrapper — YOLOv8 inference, bounding boxes, image crop |
| `db.py` | SQLite helpers — `users` + `rolls` tables, CRUD, reports |

## Environment Variables

Defined in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RESULT_FOLDER` | `results` | Output images/GIFs saved under `static/` |
| `MODEL_FOLDER` | `tflite1/custom_model_lite` | Path to TFLite model directory |
| `CAMERA_ROI` | _(empty)_ | Crop region `x,y,w,h` at 1920×1080 |
| `MAX_RETRIES` | `3` | Retries for advantage/disadvantage mode |
| `SECRET_KEY` | `dev-secret-change-me` | Flask session secret — change in production |
| `ADMIN_PASSWORD` | _(empty)_ | Password for `/admin` UI; empty = no login required |

## Database

SQLite at `dice.db` (gitignored). Schema managed by `db.init_db()` with auto-migration.

**`users` table:**
```sql
id TEXT PRIMARY KEY, enabled INTEGER DEFAULT 1, daily_limit INTEGER DEFAULT 0,
token TEXT, role TEXT DEFAULT 'user', created_at TEXT
```
- `local` user is always created with `role='admin'`, no token (used by the local UI)
- External users have a 32-char hex token generated at creation (shown once)
- `role`: `'user'` (roll + report only) or `'admin'` (full access)

**`rolls` table:**
```sql
id TEXT PRIMARY KEY, user_id TEXT, mode TEXT, detections TEXT (JSON),
selected_face INTEGER, image_path TEXT, gif_path TEXT,
time_elapsed REAL, time_elapsed_detection REAL, created_at TEXT,
reported_wrong INTEGER DEFAULT 0, correct_faces TEXT (JSON), reviewed INTEGER DEFAULT 0
```

## Roll Queue

Rolls execute one at a time via an in-memory `RollQueue` (background worker thread). The API is **async**: submit a roll, then poll or stream for the result.

**Roll flow:**
1. `POST /api/roll` (or `/u/<username>/roll`) → returns `{roll_id, position, queue_length}` immediately
2. Open SSE stream `GET /api/roll/<roll_id>/stream` for live updates, OR poll `GET /api/roll/<roll_id>/status`
3. SSE/poll responses: `{status: "queued", position: N}` → `{status: "rolling"}` → `{status: "done", result: {...}}`
4. On `done`, consume `result` (same shape as the final roll response below)

**Spam prevention:** each `client_id` gets exactly one queue slot. Re-submitting with the same `client_id` returns the existing `roll_id` and current position.

## API Reference

Base URL: `http://<pi-ip>:5000`

### Roll — no auth required

#### `POST /api/roll`
Submit a roll to the local queue.

**Body** (JSON or form-data):
```json
{ "mode": "normal", "debug": false, "client_id": "<uuid>" }
```
- `mode`: `"normal"` | `"advantage"` | `"disadvantage"` (default `"normal"`)
- `debug`: if truthy, generate a GIF
- `client_id`: browser-generated UUID stored in `localStorage`; controls queue dedup

**Response `200`:**
```json
{ "roll_id": "uuid", "position": 0, "queue_length": 1 }
```

#### `GET /api/roll/<roll_id>/stream`
SSE stream for live queue/roll status. Keep-alive until `status: done` or `unknown`.

**Event data shapes:**
```json
{ "status": "queued",  "position": 2, "queue_length": 3 }
{ "status": "rolling", "position": 0, "queue_length": 1 }
{ "status": "done",    "result": { ...roll result... }, "http_status": 200 }
{ "status": "unknown" }
```

#### `GET /api/roll/<roll_id>/status`
Polling fallback — same response shapes as SSE above.

#### Roll result object (inside `done.result`):
```json
{
  "roll_id": "uuid",
  "mode": "normal",
  "detections": [
    { "face": 15, "confidence": 0.87, "bbox": [x1, y1, x2, y2] },
    { "face": 7,  "confidence": 0.92, "bbox": [x1, y1, x2, y2] }
  ],
  "selected": { "face": 15, "confidence": 0.87, "bbox": [x1, y1, x2, y2] },
  "image": "static/results/uuid_crop.jpg",
  "gif":   "static/results/uuid.gif",
  "time_elapsed": 2.45,
  "time_elapsed_detection": 0.18
}
```
- All modes retry up to `MAX_RETRIES` times if the required number of dice aren't detected
- `normal` mode: needs ≥1 detection; `selected` = highest confidence; image cropped around bbox. Returns empty detections (no error) if nothing found after all retries
- `advantage` mode: needs ≥2 detections; `selected` = higher face value; full image
- `disadvantage` mode: needs ≥2 detections; `selected` = lower face value; full image
- If `advantage`/`disadvantage` fails to find 2 dice after `MAX_RETRIES` attempts → `http_status: 422`, `result.error: "could_not_detect_two_dice"`

#### `POST /api/roll/<roll_id>/report`
Report wrong detection(s). One value per detected die.

**Body:** `{ "correct_faces": [15, 7] }`

**Response:** `200 {"status": "ok"}` | `400` | `404`

---

### User Roll API — Bearer token required

Pass `Authorization: Bearer <token>` header. Token shown once at user creation.

#### `POST /u/<username>/roll`
Same body/response as `/api/roll`. Also enforces:
- `enabled` flag → `403 {"error": "user_disabled"}`
- `daily_limit` (0 = unlimited) → `403 {"error": "daily_limit_reached", "limit": N, "used": N}`
- Token mismatch → `401`

#### `GET /u/<username>/history?limit=20`
Returns array of roll records for that user.

---

### User Management — admin only

Admin = session cookie (UI login) or Bearer token with `role: "admin"`.

#### `GET /api/users`
List all users.

#### `POST /api/users`
Create a user. Returns token (shown only once).

**Body:** `{ "id": "alice", "daily_limit": 10, "role": "user" }`

**Response `201`:** `{ "id": "alice", "token": "<32-char hex>", "daily_limit": 10, "role": "user" }`

#### `GET /api/users/<username>`
User info + `rolls_today` count.

#### `PUT /api/users/<username>`
Update user. Accepted fields: `enabled` (bool), `daily_limit` (int), `role` (`"user"` | `"admin"`).

---

### Reports — admin only

#### `POST /api/roll/<roll_id>/acknowledge`
Mark a wrong-roll report as reviewed. **Response:** `{"status": "ok"}`

---

### UI Pages

| Path | Description |
|------|-------------|
| `GET /` | Roll page (JS-driven, uses queue) |
| `GET /admin` | User management, roll history, reports review |
| `GET /admin/login` | Login form (only needed if `ADMIN_PASSWORD` is set) |
| `GET /admin/logout` | Clear admin session |
| `GET /calibrate` | Camera ROI tool |
| `GET /label` | Training data labeling tool |
| `GET /routes` | Full interactive API reference |

---

## Model

- `tflite1/custom_model_lite/detect.tflite` — active TFLite model (swap to update)
- Labels hardcoded as `['1'..'20']` in `object_detection.py` — no labelmap file needed
- Architecture: **YOLOv8n**
- Output: `[1, 24, 8400]` → transposed to `[8400, 24]` (4 bbox + 20 class scores)
- Bounding box coordinates are **normalized (0–1)** — scale by `imW`/`imH` directly
- Input dtype: `float32`, normalized to 0–1

## Training Pipeline (PC with GPU)

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

## Camera

- Capture resolution: 1920×1080 MJPG
- Camera is kept open persistently (`dice._cap`) to eliminate per-roll init overhead (~0.5–1s)
- Motion detection runs on 480×270 downscale for Pi performance
- ROI crop configured via `CAMERA_ROI` env var (set in `/calibrate` UI)
- `fixcam.sh` locks white balance, sharpness, and power line frequency for consistent captures

## Hardware Dependencies

`RPi.GPIO` and `cv2` (OpenCV) are required at runtime on Raspberry Pi. The app will fail on import if GPIO is unavailable (non-Pi development requires mocking or skipping hardware paths).
