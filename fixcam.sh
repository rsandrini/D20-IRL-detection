#!/bin/bash
# -------------------------
# Script by Sandrini
# Configure webcam for D20 detection
# Tuned for WEBCAM PICHAU INDUS (fixed-focus, no exposure control)
# -------------------------

# Lock white balance to avoid color shifts under artificial light
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=0

# Max sharpness for cleaner dice face edges
v4l2-ctl -d /dev/video0 --set-ctrl=sharpness=2048

# Disable backlight compensation (dice box has controlled lighting)
v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=0

# Set 50Hz power line frequency (Brazil uses 60Hz — change to 2 if flickering under artificial light)
v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=2
