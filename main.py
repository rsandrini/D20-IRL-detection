import os
import uuid
from dice import *
from flask import Flask, request, jsonify
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
    # generate a new UUID for the request
    request_uuid = str(uuid.uuid4())

    # Call the roll dice method
    # Get the image, predict and return
    roll_dice(request_uuid, RESULT_FOLDER)
    detection = detector.detect_objects(f"{RESULT_FOLDER}/{request_uuid}.png")

    return jsonify({"detections":  detection,
                    "image": f"{RESULT_FOLDER}/{request_uuid}.png",
                    "gif": f"{RESULT_FOLDER}/{request_uuid}.gif"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
