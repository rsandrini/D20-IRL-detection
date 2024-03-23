from io import BytesIO

import cv2
import numpy as np
import imageio
import RPi.GPIO as GPIO
import time


def hardware_activation():
    # Pin Definitions
    pin = 6  # GPIO 6
    roll_for = 0.3

    GPIO.setmode(GPIO.BCM)  # BCM is the Broadcom SOC channel designation for GPIO numbering
    GPIO.setup(pin, GPIO.OUT)  # Set pin as an output pin
    try:
        #Turn on the GPIO pin
        GPIO.output(pin, GPIO.HIGH)
        print(f"GPIO {pin} is ON")
        time.sleep(roll_for)  # Wait for 5 seconds

        # Turn off the GPIO pin
        GPIO.output(pin, GPIO.LOW)
        print(f"GPIO {pin} is OFF")
    finally:
        GPIO.cleanup()  # Clean up GPIO settings


def roll_dice(uuid, folder):
    cap = cv2.VideoCapture(0)
    last_mean = 0
    frames_recorded = 0  # Count of frames recorded for GIF
    frame_skip = 1  # Number of frames to skip between recordings
    motion_frame_count = 0  # Count of frames with motion
    frames_since_last_motion = 0  # Count of frames since the last motion detection
    last_frame = None
    frames = []
    hardware_activation()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        last_mean = np.mean(gray)

        if frames_recorded % frame_skip == 0:
            # Reduce resolution
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(last_frame, (320, 240))  # Adjust resolution as needed
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frames_recorded += 1
        frames_since_last_motion += 1

        if result > 0.1:
            motion_frame_count += 1
            frames_since_last_motion = 0  # Reset the counter for frames since the last motion detection
        else:
            motion_frame_count = 0

        # Stop recording when motion stops for at least X frames,
        # and continue recording if frames are still being detected since the last motion
        if motion_frame_count == 0 and frames_since_last_motion >= 5:
            print(f"Motion stopped with {len(frames)} frames detected.")
            break

    # Convert frames to GIF using imageio
    duration_per_frame = 0.03  # total_duration / len(frames)  # Decreased duration per frame

    # Save the last frame as an image
    if len(frames) > 0:
        print(f"Saving image")
        cv2.imwrite(f'{folder}/{uuid}.png', cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        print(f"Last frame saved as {folder}/{uuid}.png")

    # Adjust the size and quality of the GIF
    print(f"Saving GIF in memory")
    # imageio.mimsave(f'{folder}/{uuid}.gif', frames, duration=duration_per_frame, fps=15, palettesize=5)
    # Save the GIF in memory
    gif_bytes = BytesIO()
    imageio.mimwrite(gif_bytes, frames, format='gif', duration=duration_per_frame, fps=15, palettesize=5)

    print("Finishing...")
    cap.release()
    print("Camera released")
    return last_frame, gif_bytes
