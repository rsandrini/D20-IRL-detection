"""
ROI selector — run this to visually pick the camera crop area.

Usage:
    python3 scripts/select_roi.py

Click and drag to select the dice area, then press SPACE or ENTER to confirm.
Press C to cancel and retry. The script prints the CAMERA_ROI value to paste into .env.
"""

import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Capturing frame... aim camera at the dice area first.")
for _ in range(10):  # flush buffer
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read from camera.")
    exit(1)

print("Select the dice area with your mouse. Press SPACE/ENTER to confirm, C to retry.")
roi = cv2.selectROI("Select dice area (SPACE to confirm)", frame, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = roi
if w == 0 or h == 0:
    print("No area selected.")
else:
    print(f"\nPaste this into your .env file:")
    print(f"CAMERA_ROI={x},{y},{w},{h}")
