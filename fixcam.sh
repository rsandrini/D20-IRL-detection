#!/bin/bash
# -------------------------
# Script by Sandrini
# Fix auto focus and exposure in webcam
# -------------------------
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
