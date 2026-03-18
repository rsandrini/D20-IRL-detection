import base64
import os
import uuid
import requests
import glob
import shutil
import threading
import time as _time
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, Response
from object_detection import ObjectDetector
from dice import *

TO_LABEL_DIR = "to_label"
TRAINING_DIR = "dice_training"

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


def _pending_images():
    os.makedirs(TO_LABEL_DIR, exist_ok=True)
    return sorted(glob.glob(os.path.join(TO_LABEL_DIR, "*.jpg")))


def _save_pascal_voc(filename, img_w, img_h, boxes):
    """Save Pascal VOC XML annotation matching existing training data format."""
    ann = Element("annotation")
    SubElement(ann, "folder").text = "dice_training"
    SubElement(ann, "filename").text = filename
    SubElement(ann, "path").text = os.path.join(TRAINING_DIR, filename)
    src = SubElement(ann, "source")
    SubElement(src, "database").text = "Unknown"
    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(img_w)
    SubElement(size, "height").text = str(img_h)
    SubElement(size, "depth").text = "3"
    SubElement(ann, "segmented").text = "0"

    for box in boxes:
        obj = SubElement(ann, "object")
        SubElement(obj, "name").text = str(box["label"])
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"
        bnd = SubElement(obj, "bndbox")
        SubElement(bnd, "xmin").text = str(box["x1"])
        SubElement(bnd, "ymin").text = str(box["y1"])
        SubElement(bnd, "xmax").text = str(box["x2"])
        SubElement(bnd, "ymax").text = str(box["y2"])

    xml_str = minidom.parseString(tostring(ann)).toprettyxml(indent="\t")
    xml_filename = os.path.splitext(filename)[0] + ".xml"
    with open(os.path.join(TRAINING_DIR, xml_filename), "w") as f:
        f.write(xml_str)


@app.route('/to_label/<filename>')
def to_label_file(filename):
    return send_from_directory(TO_LABEL_DIR, filename)


@app.route('/label')
def label_index():
    pending = _pending_images()
    if not pending:
        return render_template('label.html', filename=None, total=0, remaining=0)
    return redirect(url_for('label_image', filename=os.path.basename(pending[0])))


@app.route('/label/<filename>', methods=['GET'])
def label_image(filename):
    pending = _pending_images()
    total_labeled = len(glob.glob(os.path.join(TRAINING_DIR, "*.jpg")))
    return render_template('label.html',
                           filename=filename,
                           total=total_labeled + len(pending),
                           remaining=len(pending),
                           box_size=int(os.getenv('LABEL_BOX_SIZE', 100)))


@app.route('/label/<filename>', methods=['POST'])
def label_save(filename):
    data = request.get_json()
    src_path = os.path.join(TO_LABEL_DIR, filename)
    dst_path = os.path.join(TRAINING_DIR, filename)

    if data.get('skip'):
        os.remove(src_path)
    else:
        boxes = data.get('boxes', [])
        if boxes:
            img = cv2.imread(src_path)
            h, w = img.shape[:2]
            os.makedirs(TRAINING_DIR, exist_ok=True)
            shutil.move(src_path, dst_path)
            _save_pascal_voc(filename, w, h, boxes)

    pending = _pending_images()
    if pending:
        return jsonify({"next": url_for('label_image', filename=os.path.basename(pending[0]))})
    return jsonify({"next": url_for('label_index')})


def _camera_stream():
    """MJPEG stream for live calibration preview."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route('/calibrate')
def calibrate():
    return render_template('calibrate.html', roi=os.getenv('CAMERA_ROI', ''))


@app.route('/calibrate/stream')
def calibrate_stream():
    return Response(_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/calibrate/snapshot')
def calibrate_snapshot():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return '', 503
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/calibrate/save', methods=['POST'])
def calibrate_save():
    roi = request.json.get('roi', '')
    env_path = os.path.join(os.path.dirname(__file__), '.env')

    # Read existing .env
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()

    # Update or append CAMERA_ROI
    key = 'CAMERA_ROI'
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(key + '=') or line.startswith(key + ' ='):
            lines[i] = f"{key}='{roi}'\n"
            updated = True
            break
    if not updated:
        lines.append(f"{key}='{roi}'\n")

    with open(env_path, 'w') as f:
        f.writelines(lines)

    # Reload in current process
    os.environ['CAMERA_ROI'] = roi
    import dice
    dice.CAMERA_ROI = dice._parse_roi()

    return jsonify({'status': 'ok', 'roi': roi})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
