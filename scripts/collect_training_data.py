"""
Auto-roll collection script for training data.

Usage:
    python3 scripts/collect_training_data.py --count 80

Rolls the dice N times, saves each captured frame to to_label/.
No delay between rolls — moves immediately when motion stops.
Then label them at http://<pi-ip>:5000/label
"""

import argparse
import os
import sys
import time
import uuid

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dice import hardware_activation, _apply_roi, CAMERA_ROI

OUTPUT_DIR = "to_label"


def capture_after_roll():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    for _ in range(5):
        cap.read()

    hardware_activation()

    last_gray = None
    frames_since_last_motion = 0
    detection_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = _apply_roi(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_gray is not None:
            diff = cv2.absdiff(gray, last_gray)
            if np.mean(diff) > 1.5:
                frames_since_last_motion = 0
            else:
                frames_since_last_motion += 1

            detection_frame = frame.copy()

            if frames_since_last_motion >= 10:
                break

        last_gray = gray

    cap.release()
    return detection_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=80, help='Number of rolls to collect')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Collecting {args.count} rolls into '{OUTPUT_DIR}/'")
    print("Press Ctrl+C to stop early.\n")

    for i in range(args.count):
        print(f"Roll {i+1}/{args.count}...", end=" ", flush=True)
        frame = capture_after_roll()

        filename = f"roll_{i+1:04d}_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
        print(f"saved {filename}")

    print(f"\nDone. {args.count} images saved to '{OUTPUT_DIR}/'")
    print(f"Label them at http://<pi-ip>:5000/label")


if __name__ == "__main__":
    main()
