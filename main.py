import os
from dice import *

from flask import Flask, request, jsonify
from object_detection import ObjectDetector

app = Flask(__name__)
detector = ObjectDetector('tflite1/custom_model_lite')


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the D20 IRL roll page, go to /roll to roll the dice!"


@app.route('/roll', methods=['GET'])
def page_roll_dice():
    # Call the roll dice method
    # Get the image, predict and return
    roll_dice()
    detector.detect_objects("./last_frame.png")

    return jsonify({"detections":  [['20', 99.37055706977844], ['4', 96.37793302536011]],
                    "image": "last_frame.png",
                    "gif": "output.gif"})


if __name__ == '__main__':
    app.run(debug=True)