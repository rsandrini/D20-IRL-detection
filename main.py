import base64
import os
import uuid
import requests
from flask import Flask, request, render_template, jsonify
from flask_session import Session
from flask_session_captcha import FlaskSessionCaptcha
from object_detection import ObjectDetector
import redis
from dice import *

#load the .env file
from dotenv import load_dotenv
load_dotenv()

#get the envvar for the folder
MODEL_FOLDER = os.getenv("MODEL_FOLDER")
RESULT_FOLDER = os.path.join("static", os.getenv("RESULT_FOLDER"))

app = Flask(__name__)
app.config["SECRET_KEY"] = uuid.uuid4().hex

# captcha configs:
app.config['CAPTCHA_ENABLE'] = True
app.config['CAPTCHA_LENGTH'] = 5
app.config['CAPTCHA_WIDTH'] = 200
app.config['CAPTCHA_HEIGHT'] = 160

conn = redis.from_url(os.getenv('REDIS_URL'))

app.config['SESSION_TYPE'] = 'redis' # or other type of drivers for session, see https://flask-session.readthedocs.io/en/latest/
app.config['SESSION_REDIS'] = conn
Session(app)
captcha = FlaskSessionCaptcha(app)


detector = ObjectDetector(MODEL_FOLDER)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def index():
    return render_template('about.html')

@app.route('/roll-dice', methods=['GET', 'POST'])
def page_roll_dice():
    if (request.method == 'GET'):
        return render_template('roll.html')

    if request.method == 'POST' and not captcha.validate():
        return render_template('roll.html', error_message="invalid captcha")

    #TODO: calling local api now, improve this !
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
                               time_elapsed_detection=0,
                               error_message=e)

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
        if len(detection) == 1:
            detection = f"{detection[0][0]}"
        else:
            detection = f"{detection[0][0]} and {detection[1][0]}"
    except :
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
