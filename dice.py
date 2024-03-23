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


def save_frames(frames, folder):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    image_list = []
    # Save frames as JPG files
    for i, frame in enumerate(frames):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(os.path.join(folder, f"frame_{i}.jpg"))
        image_list.append(f"frame_{i}.jpg")
    return image_list


def generate_gif_from_images(load_folder, save_folder, image_list, uuid):

    images = []
    for image_file in image_list:
        image_path = os.path.join(load_folder, image_file)
        image = Image.open(image_path)
        images.append(image)

    # Save the GIF
    output_gif_path = os.path.join(save_folder, f'{uuid}.gif')
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=10,  # in milliseconds
        loop=0
    )

    print(f"GIF saved at: {output_gif_path}")


def roll_dice(uuid, folder):
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
            if motion_frame_count == 0 and frames_since_last_motion >= 10:
                print(f"Motion stopped with {len(frames)} frames detected.")

                cv2.imwrite(f'{folder}/{uuid}.jpg', cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

                start = time.time()
                # Save frames as JPG files
                temp_folder = f'{folder}/temp'
                image_list= save_frames(frames, temp_folder)

                # Create GIF from images
                generate_gif_from_images(temp_folder, folder, image_list, uuid)
                print(f"Time taken to create GIF: {time.time() - start:.2f} seconds")

                # clean up temp folder
                # for file in os.listdir(temp_folder):
                #     os.remove(os.path.join(temp_folder, file))

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
