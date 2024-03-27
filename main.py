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


@app.route('/', methods=['GET', 'POST'])
def page_roll_dice():
    if request.method == 'GET':
        return render_template('roll.html')

    #lets count the elapsed time for the roll
    roll_response = requests.post('http://localhost:5000/api/roll', data=request.form)

    # Extract data from the response
    try:
        roll_data = roll_response.json()
    except Exception as e:
        print(e)
        return render_template('roll.html',
                               gif=None,
                               result_image=None,
                               detection_text=None,
                               time_elapsed=0,
                               time_elapsed_detection=0)

    return render_template('roll.html',
                           gif=roll_data['gif'],
                           result_image=roll_data['image'],
                           detection_text=roll_data['detections'],
                           time_elapsed=roll_data['time_elapsed'],
                           time_elapsed_detection=roll_data['time_elapsed_detection'])


@app.route('/api/roll', methods=['POST'])
def api_roll_dice():
    # capture form data
    debug = request.form.get('debug', False)
    print(f"Debug: {debug}")
    start_time = time.time()
    # generate a new UUID for the request
    request_uuid = str(uuid.uuid4())

    roll_dice(request_uuid, RESULT_FOLDER, debug)

    start_time_detection = time.time()
    detection = detector.detect_objects(f"{RESULT_FOLDER}", f"{request_uuid}.jpg")
    try:
        detection = f"{detection[0][0]} and {detection[1][0]}"
    except:
        detection = "No dice detected :("
    time_elapsed_detection = round(time.time() - start_time_detection, 2)
    time_elapsed = round(time.time() - start_time, 2)
    return jsonify({"detections":  detection,
                    "image": f"{RESULT_FOLDER}/{request_uuid}.jpg",
                    "gif": f"{RESULT_FOLDER}/{request_uuid}.gif",
                    "time_elapsed": time_elapsed,
                    "time_elapsed_detection": time_elapsed_detection
                    }
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
