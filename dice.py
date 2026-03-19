import concurrent.futures
import random
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
from dotenv import load_dotenv

load_dotenv()


def _parse_roi():
    raw = os.getenv("CAMERA_ROI", "").strip()
    if not raw:
        return None
    try:
        x, y, w, h = [int(v) for v in raw.split(",")]
        return (x, y, w, h)
    except ValueError:
        print(f"WARNING: Invalid CAMERA_ROI value '{raw}', ignoring.")
        return None


CAMERA_ROI = _parse_roi()

# ---------------------------------------------------------------------------
# Persistent camera — opened once, reused across rolls to eliminate the
# ~500–1000ms VideoCapture(0) init cost that happened on every roll.
# ---------------------------------------------------------------------------
_cap = None


def _get_cap():
    """Return the shared camera, (re)initializing it if needed."""
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap
    print("Initializing camera...")
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # Drain stale frames after a fresh open
    for _ in range(5):
        _cap.read()
    return _cap


def release_camera():
    """Explicitly release the shared camera (e.g. to free it for calibration)."""
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None


def _apply_roi(frame):
    if CAMERA_ROI is None:
        return frame
    x, y, w, h = CAMERA_ROI
    return frame[y:y+h, x:x+w]


def hardware_activation():
    pin = 6  # GPIO 6

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    try:
        # Shake pattern: rapid pulses to agitate the dice
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.3)

        # Final push: random duration so dice lands differently each roll
        roll_for = random.uniform(0.2, 0.4)
        GPIO.output(pin, GPIO.HIGH)
        print(f"GPIO {pin} rolling for {roll_for:.2f}s")
        time.sleep(roll_for)
        GPIO.output(pin, GPIO.LOW)
    finally:
        GPIO.cleanup()


def process_frames(frames):
    image_list = []
    for frame in frames[-25:]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mem = BytesIO()
        Image.fromarray(frame_rgb).save(mem, format="JPEG")
        mem.seek(0)
        image_list.append(mem)
    return image_list


def generate_gif_from_images(image_list, gif_name):
    processed_images = [Image.open(mem) for mem in image_list]
    processed_images[0].save(
        gif_name,
        save_all=True,
        append_images=processed_images[1:],
        duration=10,
        loop=1
    )
    print(f"GIF saved at: {gif_name}")


def roll_dice(uuid, folder, debug):
    cap = _get_cap()

    # Flush stale frames from the buffer (camera was idle between rolls).
    # 2 frames is enough — the buffer is small on MJPG cameras.
    for _ in range(2):
        cap.read()

    last_frame_gray = None
    frames_since_last_motion = 0
    frames = []             # GIF frames (small, RGB)
    detection_frame = None  # Full-res ROI frame for model inference

    # Auto-calibrate motion threshold from baseline noise before rolling.
    # 5 frames is enough for a stable noise-floor estimate.
    print("Calibrating motion baseline...")
    baseline_scores = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            # Camera dropped — reinitialize and retry once
            release_camera()
            cap = _get_cap()
            ret, frame = cap.read()
            if not ret:
                continue
        small = cv2.resize(_apply_roi(frame), (480, 270), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if last_frame_gray is not None:
            baseline_scores.append(np.mean(cv2.absdiff(gray, last_frame_gray)))
        last_frame_gray = gray

    noise_floor = np.mean(baseline_scores) if baseline_scores else 1.0
    motion_threshold = max(noise_floor * 3, noise_floor + 2.0)
    print(f"Noise floor: {noise_floor:.2f} → motion threshold: {motion_threshold:.2f}")
    last_frame_gray = None

    hardware_activation()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = _apply_roi(frame)
        # Downscale only for motion detection — keeps the loop fast on Pi
        small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if last_frame_gray is not None:
            diff = cv2.absdiff(gray, last_frame_gray)
            motion_score = np.mean(diff)

            if motion_score > motion_threshold:
                frames_since_last_motion = 0
            else:
                frames_since_last_motion += 1

            # Keep the most recent frame for detection (before resize)
            detection_frame = frame.copy()

            # Store GIF frame (small)
            gif_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            gif_frame = cv2.cvtColor(gif_frame, cv2.COLOR_BGR2RGB)
            frames.append(gif_frame)

            if frames_since_last_motion >= 10:
                print(f"Motion stopped with {len(frames)} frames detected.")
                break

        last_frame_gray = gray  # small (480x270) for fast comparison

    # Save detection frame at full ROI resolution (not the tiny GIF size)
    cv2.imwrite(f'{folder}/{uuid}.jpg', detection_frame)

    if debug and len(frames) > 1:
        start = time.time()
        generate_gif_from_images(process_frames(frames), f'{folder}/{uuid}.gif')
        print(f"Time taken to create GIF: {time.time() - start:.2f} seconds")

    print("Finishing...")
    # Camera stays open — no cap.release()


if __name__ == "__main__":
    RESULT_FOLDER = "results"
    uuid = "unique_id"
    roll_dice(uuid, RESULT_FOLDER, debug=False)
    release_camera()
