import base64
import json
import os
import uuid
import glob
import shutil
import threading
import time as _time
from functools import wraps
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, Response, session, stream_with_context, g
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
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
detector = ObjectDetector(MODEL_FOLDER)
db.init_db()

# Print local token to console for first-time setup
_local = db.get_user("local")
if _local and _local["token"]:
    print(f"\n  Local user token: {_local['token']}\n")



# ---------------------------------------------------------------------------
# RollQueue — single-threaded roll executor
# ---------------------------------------------------------------------------

class RollQueue:
    """Single-threaded roll executor. Each client_id gets one queue slot."""

    _MAX_RESULTS = 200

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = []      # [{client_id, roll_id, params}]
        self._active  = None    # current item being executed
        self._results = {}      # roll_id -> {data, http_status}
        self._work    = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, client_id, params):
        """Returns (roll_id, position). Position 0 = will roll next, 1+ = waiting."""
        with self._lock:
            if self._active and self._active["client_id"] == client_id:
                return self._active["roll_id"], 0
            for i, item in enumerate(self._pending):
                if item["client_id"] == client_id:
                    offset = 1 if self._active else 0
                    return item["roll_id"], i + offset
            roll_id = str(uuid.uuid4())
            offset  = 1 if self._active else 0
            pos     = len(self._pending) + offset
            self._pending.append({"client_id": client_id, "roll_id": roll_id, "params": params})
        self._work.set()
        return roll_id, pos

    def status(self, roll_id):
        with self._lock:
            if roll_id in self._results:
                r = self._results[roll_id]
                return {"status": "done", "result": r["data"], "http_status": r["http_status"]}
            if self._active and self._active["roll_id"] == roll_id:
                return {"status": "rolling", "position": 0, "queue_length": self._ql()}
            for i, item in enumerate(self._pending):
                if item["roll_id"] == roll_id:
                    offset = 1 if self._active else 0
                    return {"status": "queued", "position": i + offset, "queue_length": self._ql()}
        return {"status": "unknown"}

    def queue_length(self):
        with self._lock:
            return self._ql()

    def _ql(self):
        return len(self._pending) + (1 if self._active else 0)

    def _loop(self):
        while True:
            self._work.wait()
            self._work.clear()
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    item = self._pending[0]
                    self._active = item
                try:
                    data, http_status = _do_roll(**item["params"])
                except Exception as e:
                    data, http_status = {"error": str(e)}, 500
                with self._lock:
                    self._pending.pop(0)
                    self._results[item["roll_id"]] = {"data": data, "http_status": http_status}
                    self._active = None
                    if len(self._results) > self._MAX_RESULTS:
                        oldest = next(iter(self._results))
                        del self._results[oldest]


roll_queue = RollQueue()

# Start queue worker (skip in werkzeug reloader child to avoid double-start)
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    roll_queue.start()


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


def _resolve_user():
    """Resolve caller to a user dict, or None if unauthenticated.
    Checks: Bearer token header → ?token query param → LOCAL_SUBNET IP bypass → admin session.
    """
    # 1. Bearer token in Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user = db.get_user_by_token(auth[7:].strip())
        if user and user["enabled"]:
            return dict(user)

    # 2. Token in query string (required for EventSource which cannot set headers)
    token = request.args.get("token", "").strip()
    if token:
        user = db.get_user_by_token(token)
        if user and user["enabled"]:
            return dict(user)

    # 3. IP subnet bypass — LOCAL_SUBNET env var (e.g. "192.168.1.0/24")
    subnet_str = os.getenv("LOCAL_SUBNET", "").strip()
    if subnet_str:
        try:
            from ipaddress import ip_network, ip_address
            if ip_address(request.remote_addr) in ip_network(subnet_str, strict=False):
                user = db.get_user("local")
                if user and user["enabled"]:
                    return dict(user)
        except ValueError:
            pass

    # 4. Admin session (browser login via /admin/login — only for admin UI pages)
    if session.get("is_admin") or not os.getenv("ADMIN_PASSWORD"):
        user = db.get_user("local")
        if user and user["enabled"]:
            return dict(user)

    return None


def require_auth(f):
    """Require any valid authenticated user. Sets g.current_user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = _resolve_user()
        if not user:
            return jsonify({"error": "unauthorized"}), 403
        if not user.get("enabled"):
            return jsonify({"error": "user_disabled"}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Require admin role. Sets g.current_user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = _resolve_user()
        if not user:
            if request.is_json or request.accept_mimetypes.best == "application/json":
                return jsonify({"error": "unauthorized"}), 403
            return redirect(url_for("admin_login"))
        if user.get("role") != "admin":
            if request.is_json or request.accept_mimetypes.best == "application/json":
                return jsonify({"error": "admin_required"}), 403
            return redirect(url_for("admin_login"))
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def _blur_nonselected(frames, detections, selected, det_shape):
    """Blur non-selected dice regions in all GIF frames.
    Frames are 320x240 RGB; bboxes are in detection-frame pixel coords."""
    if not detections or not selected or not det_shape or len(detections) < 2:
        return frames
    H_det, W_det = det_shape[:2]
    if W_det == 0 or H_det == 0:
        return frames
    x_scale = 320 / W_det
    y_scale = 240 / H_det
    blur_bboxes = [
        [max(0, int(d["bbox"][0] * x_scale)), max(0, int(d["bbox"][1] * y_scale)),
         min(320, int(d["bbox"][2] * x_scale)), min(240, int(d["bbox"][3] * y_scale))]
        for d in detections if d is not selected and d.get("bbox")
    ]
    if not blur_bboxes:
        return frames
    result = []
    for frame in frames:
        f = frame.copy()
        for x1, y1, x2, y2 in blur_bboxes:
            if x2 > x1 and y2 > y1:
                f[y1:y2, x1:x2] = cv2.GaussianBlur(f[y1:y2, x1:x2], (51, 51), 0)
        result.append(f)
    return result


def _do_roll(user_id, mode, debug):
    """Core roll + detect logic. Returns (response_dict, http_status)."""
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    start_time = _time.time()
    request_uuid = str(uuid.uuid4())
    time_elapsed_detection = 0
    detections = []
    frames, det_shape = [], None

    need = 2 if mode in ("advantage", "disadvantage") else 1
    got_enough = False

    for _ in range(max_retries):
        frames, det_shape = roll_dice(request_uuid, RESULT_FOLDER)
        if detector.interpreter is None:
            got_enough = True   # no model — don't retry, just proceed
            break
        det_start = _time.time()
        detections = detector.detect_objects(RESULT_FOLDER, f"{request_uuid}.jpg")
        time_elapsed_detection += round(_time.time() - det_start, 4)
        if len(detections) >= need:
            got_enough = True
            break

    if not got_enough and detector.interpreter is not None:
        if mode in ("advantage", "disadvantage"):
            time_elapsed = round(_time.time() - start_time, 2)
            return {"error": "could_not_detect_two_dice", "detections": detections,
                    "time_elapsed": time_elapsed}, 422
        # normal mode: no dice after all retries — return empty result, don't hard-fail

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
    if debug and len(frames) > 1:
        t = _time.time()
        gif_frames = _blur_nonselected(frames, detections, selected, det_shape)
        generate_gif_from_images(process_frames(gif_frames), gif_path)
        print(f"GIF generated in {_time.time() - t:.2f}s")

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
    if request.method == 'POST':
        return redirect(url_for('page_roll_dice'))
    return render_template('roll.html')


# ---------------------------------------------------------------------------
# /api/roll — async via queue
# ---------------------------------------------------------------------------

@app.route('/api/roll', methods=['POST'])
@require_auth
def api_roll_dice():
    client_id = _get_param("client_id") or request.headers.get("X-Client-Id") or g.current_user["id"]
    mode  = _get_param("mode", "normal")
    debug = bool(_get_param("debug", False))
    roll_id, position = roll_queue.submit(client_id, {
        "user_id": g.current_user["id"],
        "mode": mode,
        "debug": debug,
    })
    return jsonify({"roll_id": roll_id, "position": position, "queue_length": roll_queue.queue_length()})


@app.route('/api/roll/<roll_id>/stream')
@require_auth
def roll_stream(roll_id):
    def generate():
        while True:
            s = roll_queue.status(roll_id)
            yield f"data: {json.dumps(s)}\n\n"
            if s["status"] in ("done", "unknown"):
                return
            _time.sleep(0.5)
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/api/roll/<roll_id>/status')
@require_auth
def roll_status(roll_id):
    return jsonify(roll_queue.status(roll_id))


# ---------------------------------------------------------------------------
# /api/roll/<roll_id>/report
# ---------------------------------------------------------------------------

@app.route('/api/roll/<roll_id>/report', methods=['POST'])
@require_auth
def report_roll(roll_id):
    data = request.get_json(silent=True) or {}
    raw = data.get("correct_faces")
    if raw is None:
        return jsonify({"error": "correct_faces required"}), 400
    if not isinstance(raw, list):
        raw = [raw]
    try:
        correct_faces = [int(v) for v in raw]
        if not all(1 <= v <= 20 for v in correct_faces):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "correct_faces must be list of ints 1–20"}), 400

    changed = db.report_roll(roll_id, correct_faces)
    if not changed:
        return jsonify({"error": "roll not found"}), 404
    return jsonify({"status": "ok"})


@app.route('/api/roll/<roll_id>/acknowledge', methods=['POST'])
@admin_required
def acknowledge_report(roll_id):
    db.acknowledge_report(roll_id)
    return jsonify({"status": "ok"})


@app.route('/api/auth/me')
@require_auth
def auth_me():
    u = g.current_user
    return jsonify({"id": u["id"], "role": u["role"], "enabled": u["enabled"]})


# ---------------------------------------------------------------------------
# /u/<username>/roll  and  /u/<username>/history
# ---------------------------------------------------------------------------

@app.route('/u/<username>/roll', methods=['POST'])
@require_auth
def user_roll(username):
    # Must be rolling as yourself or be an admin
    if g.current_user["id"] != username and g.current_user.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403

    user = db.get_user(username)
    if not user:
        return jsonify({"error": "user not found"}), 404
    if not user["enabled"]:
        return jsonify({"error": "user_disabled"}), 403

    daily_limit = user["daily_limit"]
    if daily_limit > 0 and db.get_user_roll_count_today(username) >= daily_limit:
        return jsonify({"error": "daily_limit_reached"}), 403

    client_id = _get_param("client_id") or f"user:{username}"
    mode  = _get_param("mode", "normal")
    debug = bool(_get_param("debug", False))
    roll_id, position = roll_queue.submit(client_id, {
        "user_id": username,
        "mode": mode,
        "debug": debug,
    })
    return jsonify({"roll_id": roll_id, "position": position, "queue_length": roll_queue.queue_length()})


@app.route('/u/<username>/history', methods=['GET'])
@require_auth
def user_history(username):
    if g.current_user["id"] != username and g.current_user.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    limit = int(request.args.get("limit", 20))
    rolls = db.get_rolls(user_id=username, limit=limit)
    return jsonify(rolls)


# ---------------------------------------------------------------------------
# /api/users  (user management)
# ---------------------------------------------------------------------------

@app.route('/api/users', methods=['GET'])
@admin_required
def list_users():
    return jsonify(db.list_users())


@app.route('/api/users', methods=['POST'])
@admin_required
def create_user():
    data = request.get_json(silent=True) or {}
    user_id = data.get("id", "").strip()
    if not user_id:
        return jsonify({"error": "id required"}), 400
    if db.get_user(user_id):
        return jsonify({"error": "user already exists"}), 409
    daily_limit = int(data.get("daily_limit", 0))
    role = data.get("role", "user")
    token = db.create_user(user_id, daily_limit, role)
    return jsonify({"id": user_id, "token": token, "daily_limit": daily_limit, "role": role}), 201


@app.route('/api/users/<username>', methods=['GET'])
@admin_required
def get_user(username):
    user = db.get_user(username)
    if not user:
        return jsonify({"error": "not found"}), 404
    today_count = db.get_user_roll_count_today(username)
    return jsonify({**dict(user), "rolls_today": today_count})


@app.route('/api/users/<username>', methods=['PUT'])
@admin_required
def update_user(username):
    if not db.get_user(username):
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    kwargs = {}
    if "enabled" in data:
        kwargs["enabled"] = int(bool(data["enabled"]))
    if "daily_limit" in data:
        kwargs["daily_limit"] = int(data["daily_limit"])
    if "role" in data and data["role"] in ("admin", "user"):
        kwargs["role"] = data["role"]
    db.update_user(username, **kwargs)
    return jsonify(dict(db.get_user(username)))


# ---------------------------------------------------------------------------
# /admin
# ---------------------------------------------------------------------------

@app.route('/routes')
def routes():
    return render_template('routes.html')


@app.route('/admin')
@admin_required
def admin():
    import json as _json
    users = db.list_users()
    rolls = db.get_rolls(limit=50)
    for r in rolls:
        try:
            r['detections_parsed'] = _json.loads(r['detections']) if r.get('detections') else []
        except Exception:
            r['detections_parsed'] = []
    reports = db.get_reports(only_unreviewed=False)
    for r in reports:
        try:
            r['detections_parsed'] = _json.loads(r['detections']) if r.get('detections') else []
            r['correct_faces_parsed'] = _json.loads(r['correct_faces']) if r.get('correct_faces') else []
        except Exception:
            r['detections_parsed'] = []
            r['correct_faces_parsed'] = []
    pending_reports = sum(1 for r in reports if not r.get('reviewed'))
    return render_template('admin.html', users=users, rolls=rolls, reports=reports, pending_reports=pending_reports)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == os.getenv('ADMIN_PASSWORD', ''):
            session['is_admin'] = True
            return redirect(url_for('admin'))
        return render_template('admin_login.html', error='Wrong password')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('page_roll_dice'))


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
    app.run(host='0.0.0.0', debug=True, threaded=True)
