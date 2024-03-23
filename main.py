import base64
import os
import uuid
import requests
from flask import Flask, request, render_template, jsonify
from object_detection import ObjectDetector
from dice import *

#load the .env file
from dotenv import load_dotenv
load_dotenv()

#get the envvar for the folder
MODEL_FOLDER = os.getenv("MODEL_FOLDER")
RESULT_FOLDER = os.path.join("static", os.getenv("RESULT_FOLDER"))

app = Flask(__name__)
detector = ObjectDetector(MODEL_FOLDER)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/roll', methods=['GET', 'POST'])
def page_roll_dice():
    if request.method == 'GET':
        return render_template('roll.html')

    #lets count the elapsed time for the roll
    start_time = time.time()
    roll_response = requests.post('http://localhost:5000/api/roll')

    try:
        # Extract data from the response
        roll_data = roll_response.json()

        start_time_detection = time.time()
        detection_text = roll_data['detections']
        gif_base64 = roll_data['gif_base64']
        time_elapsed_detection = time.time() - start_time_detection
        time_elapsed = time.time() - start_time

        return render_template('roll.html',
                               result_gif_base64=gif_base64,
                               detection_text=detection_text,
                               time_elapsed=time_elapsed,
                               time_elapsed_detection=time_elapsed_detection)
    except Exception as e:
        raise


@app.route('/api/roll', methods=['POST'])
def api_roll_dice():
    # generate a new UUID for the request
    request_uuid = str(uuid.uuid4())

    # Call the roll dice method
    # Get the image, predict and return
    _, gif_bytes = roll_dice(request_uuid, RESULT_FOLDER)
    gif_base64 = base64.b64encode(gif_bytes.getvalue()).decode('utf-8')
    detection = detector.detect_objects(f"{RESULT_FOLDER}/{request_uuid}.png")

    return jsonify({"detections":  detection,
                    "image": f"{RESULT_FOLDER}/{request_uuid}.png",
                    "gif_base64": gif_base64
                    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
