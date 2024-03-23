import os
import uuid

import requests
from werkzeug.utils import send_from_directory

from dice import *
from flask import Flask, request, render_template, jsonify
from object_detection import ObjectDetector

#load the .env file
from dotenv import load_dotenv
load_dotenv()

#get the envvar for the folder
MODEL_FOLDER = os.getenv("MODEL_FOLDER")
RESULT_FOLDER = os.getenv("RESULT_FOLDER")


app = Flask(__name__)
detector = ObjectDetector(MODEL_FOLDER)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/roll', methods=['POST'])
def page_roll_dice():
    # Call the /roll method as an API

    #lets count the elapsed time for the roll
    start_time = time.time()
    roll_response = requests.post('http://localhost:5000/roll')

    # Extract data from the response
    roll_data = roll_response.json()
    result_gif = roll_data['gif']
    detection_text = ", ".join(roll_data['detections'])
    time_elapsed = time.time() - start_time
    return render_template('roll.html', result_gif=result_gif, detection_text=detection_text, time_elapsed=time_elapsed)


@app.route('/roll', methods=['POST'])
def api_roll_dice():
    # generate a new UUID for the request
    request_uuid = str(uuid.uuid4())

    # Call the roll dice method
    # Get the image, predict and return
    roll_dice(request_uuid, RESULT_FOLDER)
    detection = detector.detect_objects(f"{RESULT_FOLDER}/{request_uuid}.png")

    return jsonify({"detections":  detection,
                    "image": f"{RESULT_FOLDER}/{request_uuid}.png",
                    "gif": f"{RESULT_FOLDER}/{request_uuid}.gif"})


@app.route(f'/{RESULT_FOLDER}/<path:image_name>', methods=['GET'])
def get_image(image_name):
    return send_from_directory(RESULT_FOLDER, image_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
