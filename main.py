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
import db

TO_LABEL_DIR = "to_label"
TRAINING_DIR = "dice_training"

from dotenv import load_dotenv
load_dotenv()

MODEL_FOLDER = os.getenv("MODEL_FOLDER")
RESULT_FOLDER = os.path.join("static", os.getenv("RESULT_FOLDER"))

app = Flask(__name__)
detector = ObjectDetector(MODEL_FOLDER)
db.init_db()

roll_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_param(key, default=None):
    """Read a param from JSON body, form data, or query string."""
    if request.is_json:
        return request.get_json(silent=True, force=True).get(key, default)
    val = request.form.get(key)
    if val is None:
        val = request.args.get(key)
    return val if val is not None else default


def _authenticate(username):
    """Check Bearer token for external user routes. Returns (user_row, error_response)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, (jsonify({"error": "missing_token"}), 401)
    token = auth[len("Bearer "):]
    user = db.get_user_by_token(token)
    if user is None or user["id"] != username:
        return None, (jsonify({"error": "invalid_token"}), 401)
    return user, None


def _do_roll(user_id, mode, debug):
    """Core roll + detect logic. Returns (response_dict, http_status)."""
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    start_time = _time.time()
    request_uuid = str(uuid.uuid4())
    time_elapsed_detection = 0
    detections = []

    if mode in ("advantage", "disadvantage"):
        got_two = False
        for _ in range(max_retries):
            roll_dice(request_uuid, RESULT_FOLDER, debug)
            if detector.interpreter is not None:
                det_start = _time.time()
                detections = detector.detect_objects(RESULT_FOLDER, f"{request_uuid}.jpg")
                time_elapsed_detection += round(_time.time() - det_start, 4)
                if len(detections) >= 2:
                    got_two = True
                    break
        if not got_two and detector.interpreter is not None:
            time_elapsed = round(_time.time() - start_time, 2)
            return {"error": "could_not_detect_two_dice", "detections": detections,
                    "time_elapsed": time_elapsed}, 422
    else:
        roll_dice(request_uuid, RESULT_FOLDER, debug)
        if detector.interpreter is not None:
            det_start = _time.time()
            detections = detector.detect_objects(RESULT_FOLDER, f"{request_uuid}.jpg")
            time_elapsed_detection = round(_time.time() - det_start, 2)

    # Select winner
    if mode == "advantage":
        selected = max(detections[:2], key=lambda d: d["face"]) if len(detections) >= 2 else None
    elif mode == "disadvantage":
        selected = min(detections[:2], key=lambda d: d["face"]) if len(detections) >= 2 else None
    else:
        selected = max(detections, key=lambda d: d["confidence"]) if detections else None

    image_path = f"{RESULT_FOLDER}/{request_uuid}.jpg"

    # Crop for normal mode
    if mode == "normal" and selected and selected.get("bbox"):
        crop_path = detector.crop_image(image_path, selected["bbox"], margin=20)
        if crop_path:
            image_path = crop_path

    gif_path = f"{RESULT_FOLDER}/{request_uuid}.gif"
    time_elapsed = round(_time.time() - start_time, 2)
    time_elapsed_detection = round(time_elapsed_detection, 2)

    db.save_roll(
        request_uuid, user_id, mode, detections,
        selected["face"] if selected else None,
        image_path, gif_path, time_elapsed, time_elapsed_detection
    )

    return {
        "roll_id": request_uuid,
        "mode": mode,
        "detections": detections,
        "selected": selected,
        "image": image_path,
        "gif": gif_path,
        "time_elapsed": time_elapsed,
        "time_elapsed_detection": time_elapsed_detection,
    }, 200


# ---------------------------------------------------------------------------
# Local UI
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def page_roll_dice():
    if request.method == 'GET':
        return render_template('roll.html')

    roll_response = requests.post('http://localhost:5000/api/roll', data=request.form)

    try:
        roll_data = roll_response.json()
    except Exception as e:
        print(e)
        return render_template('roll.html')

    if roll_response.status_code != 200:
        return render_template('roll.html', error=roll_data.get("error"))

    return render_template('roll.html',
                           gif=roll_data.get('gif'),
                           result_image=roll_data.get('image'),
                           selected=roll_data.get('selected'),
                           detections=roll_data.get('detections', []),
                           mode=roll_data.get('mode', 'normal'),
                           roll_id=roll_data.get('roll_id'),
                           time_elapsed=roll_data.get('time_elapsed', 0),
                           time_elapsed_detection=roll_data.get('time_elapsed_detection', 0))


# ---------------------------------------------------------------------------
# /api/roll — local endpoint (no auth)
# ---------------------------------------------------------------------------

@app.route('/api/roll', methods=['POST'])
def api_roll_dice():
    if not roll_lock.acquire(blocking=False):
        return jsonify({"error": "roll_in_progress"}), 429
    try:
        mode = _get_param("mode", "normal")
        debug = _get_param("debug", False)
        result, status = _do_roll("local", mode, debug)
        return jsonify(result), status
    finally:
        roll_lock.release()


# ---------------------------------------------------------------------------
# /api/roll/<roll_id>/report
# ---------------------------------------------------------------------------

@app.route('/api/roll/<roll_id>/report', methods=['POST'])
def report_roll(roll_id):
    data = request.get_json(silent=True) or {}
    correct_face = data.get("correct_face")
    if correct_face is None:
        return jsonify({"error": "correct_face required"}), 400
    try:
        correct_face = int(correct_face)
        if not (1 <= correct_face <= 20):
            raise ValueError
    except ValueError:
        return jsonify({"error": "correct_face must be 1–20"}), 400

    changed = db.report_roll(roll_id, correct_face)
    if not changed:
        return jsonify({"error": "roll not found"}), 404
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# /u/<username>/roll  and  /u/<username>/history
# ---------------------------------------------------------------------------

@app.route('/u/<username>/roll', methods=['POST'])
def user_roll(username):
    user, err = _authenticate(username)
    if err:
        return err

    if not user["enabled"]:
        return jsonify({"error": "user_disabled"}), 403

    daily_limit = user["daily_limit"]
    if daily_limit > 0:
        count_today = db.get_user_roll_count_today(username)
        if count_today >= daily_limit:
            return jsonify({"error": "daily_limit_reached", "limit": daily_limit, "used": count_today}), 403

    if not roll_lock.acquire(blocking=False):
        return jsonify({"error": "roll_in_progress"}), 429
    try:
        mode = _get_param("mode", "normal")
        debug = _get_param("debug", False)
        result, status = _do_roll(username, mode, debug)
        return jsonify(result), status
    finally:
        roll_lock.release()


@app.route('/u/<username>/history', methods=['GET'])
def user_history(username):
    user, err = _authenticate(username)
    if err:
        return err
    limit = int(request.args.get("limit", 20))
    rolls = db.get_rolls(user_id=username, limit=limit)
    return jsonify(rolls)


# ---------------------------------------------------------------------------
# /api/users  (user management)
# ---------------------------------------------------------------------------

@app.route('/api/users', methods=['GET'])
def list_users():
    return jsonify(db.list_users())


@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get("id", "").strip()
    if not user_id:
        return jsonify({"error": "id required"}), 400
    if db.get_user(user_id):
        return jsonify({"error": "user already exists"}), 409
    daily_limit = int(data.get("daily_limit", 0))
    token = db.create_user(user_id, daily_limit)
    return jsonify({"id": user_id, "token": token, "daily_limit": daily_limit}), 201


@app.route('/api/users/<username>', methods=['GET'])
def get_user(username):
    user = db.get_user(username)
    if not user:
        return jsonify({"error": "not found"}), 404
    today_count = db.get_user_roll_count_today(username)
    return jsonify({**dict(user), "rolls_today": today_count})


@app.route('/api/users/<username>', methods=['PUT'])
def update_user(username):
    if not db.get_user(username):
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    kwargs = {}
    if "enabled" in data:
        kwargs["enabled"] = int(bool(data["enabled"]))
    if "daily_limit" in data:
        kwargs["daily_limit"] = int(data["daily_limit"])
    db.update_user(username, **kwargs)
    return jsonify(dict(db.get_user(username)))


# ---------------------------------------------------------------------------
# /admin
# ---------------------------------------------------------------------------

@app.route('/routes')
def routes():
    return render_template('routes.html')


@app.route('/admin')
def admin():
    import json as _json
    users = db.list_users()
    rolls = db.get_rolls(limit=50)
    for r in rolls:
        try:
            r['detections_parsed'] = _json.loads(r['detections']) if r.get('detections') else []
        except Exception:
            r['detections_parsed'] = []
    return render_template('admin.html', users=users, rolls=rolls)


# ---------------------------------------------------------------------------
# Labeling routes (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Calibration routes (unchanged)
# ---------------------------------------------------------------------------

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

    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()

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

    os.environ['CAMERA_ROI'] = roi
    import dice
    dice.CAMERA_ROI = dice._parse_roi()

    return jsonify({'status': 'ok', 'roi': roi})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
