import concurrent.futures
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time


def hardware_activation():
    # Pin Definitions
    pin = 6  # GPIO 6
    roll_for = 0.3

    GPIO.setmode(GPIO.BCM)  # BCM is the Broadcom SOC channel designation for GPIO numbering
    GPIO.setup(pin, GPIO.OUT)  # Set pin as an output pin
    try:
        # Turn on the GPIO pin
        GPIO.output(pin, GPIO.HIGH)
        print(f"GPIO {pin} is ON")
        time.sleep(roll_for)  # Wait for 5 seconds

        # Turn off the GPIO pin
        GPIO.output(pin, GPIO.LOW)
        print(f"GPIO {pin} is OFF")
    finally:
        GPIO.cleanup()  # Clean up GPIO settings


def process_frames(frames):
    image_list = []
    for i, frame in enumerate(frames[-25:]):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Create BytesIO object to store image in memory
        mem = BytesIO()
        # Save frame as JPEG to BytesIO object
        Image.fromarray(frame_rgb).save(mem, format="JPEG")
        # Move to the beginning of BytesIO object
        mem.seek(0)
        # Append BytesIO object to the list
        image_list.append(mem)
    return image_list


def generate_gif_from_images(image_list, gif_name):
    processed_images = [Image.open(mem) for mem in image_list]
    processed_images[0].save(
        gif_name,
        save_all=True,
        append_images=processed_images[1:],
        duration=10,  # in milliseconds
        loop=0
    )

    print(f"GIF saved at: {gif_name}")


def roll_dice(uuid, folder, debug):
    cap = cv2.VideoCapture(0)
    last_mean = 0
    frames_recorded = 0  # Count of frames recorded for GIF
    frame_skip = 1  # Number of frames to skip between recordings
    motion_frame_count = 0  # Count of frames with motion
    frames_since_last_motion = 0  # Count of frames since the last motion detection
    frames = []
    hardware_activation()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = np.abs(np.mean(gray) - last_mean)
            last_mean = np.mean(gray)

            if frames_recorded % frame_skip == 0:
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frames_recorded += 1
            frames_since_last_motion += 1

            if result > 0.1:
                motion_frame_count += 1
                frames_since_last_motion = 0  # Reset the counter for frames since the last motion detection
            else:
                motion_frame_count = 0

            # Stop recording when motion stops for at least 10 frames,
            # and continue recording if frames are still being detected since the last motion
            if motion_frame_count == 0 and frames_since_last_motion >= 7:
                print(f"Motion stopped with {len(frames)} frames detected.")

                cv2.imwrite(f'{folder}/{uuid}.jpg', cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
                if debug:
                    start = time.time()
                    # Create GIF from images
                    generate_gif_from_images(process_frames(frames), f'{folder}/{uuid}.gif')
                    print(f"Time taken to create GIF: {time.time() - start:.2f} seconds")
                break

    print("Finishing...")
    cap.release()
    print("Camera released")


if __name__ == "__main__":
    # Define folder to save results
    RESULT_FOLDER = "results"

    # Define a unique identifier for the roll
    uuid = "unique_id"

    # Roll the dice and save results
    roll_dice(uuid, RESULT_FOLDER)
