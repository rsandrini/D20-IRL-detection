#!/bin/bash
# -------------------------
# Script by Sandrini
# Fix auto focus and exposure in webcam
# -------------------------
# Fix the focus for the distance
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=520

# Fix zoom and exposure
v4l2-ctl -d /dev/video0 --set-ctrl=zoom_absolute=100
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
