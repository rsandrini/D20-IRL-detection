######## Webcam Object Detection Using YOLOv8 TFLite Model #########
#
# Originally by Evan Juras, updated for YOLOv8n TFLite (D20 project)
#
# Description:
# Runs YOLOv8n TFLite inference on a live webcam feed, drawing bounding
# boxes and labels for detected D20 faces (1–20) in real time.
# The webcam runs in a separate thread to improve FPS.
#
# YOLOv8 TFLite output format: [1, 4+nc, num_anchors]
# Transposed to [num_anchors, 4+nc]: first 4 cols = cx,cy,w,h; rest = class scores.

import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread

class VideoStream:
    """Captures frames from a webcam in a background thread."""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder containing detect.tflite', required=True)
parser.add_argument('--graph', help='.tflite filename', default='detect.tflite')
parser.add_argument('--threshold', help='Minimum confidence threshold', type=float, default=0.5)
parser.add_argument('--nms_threshold', help='NMS IoU threshold', type=float, default=0.45)
parser.add_argument('--resolution', help='Webcam resolution WxH', default='1280x720')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
min_conf_threshold = args.threshold
nms_threshold = args.nms_threshold
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Labels are hardcoded for D20 (faces 1–20)
labels = [str(i) for i in range(1, 21)]

# Import TFLite interpreter
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

# Load model
PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, GRAPH_NAME)
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width  = input_details[0]['shape'][2]
is_quantized = input_details[0]['dtype'] == np.uint8

# FPS tracking
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Start video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

while True:
    t1 = cv2.getTickCount()

    frame = videostream.read().copy()

    # Preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if is_quantized:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data.astype(np.float32) / 255.0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # YOLOv8 output: [1, 4+nc, num_anchors] → transpose → [num_anchors, 4+nc]
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index'])).T
    boxes_raw   = output[:, :4]     # cx, cy, w, h in model input coords
    class_scores = output[:, 4:]    # per-class confidence scores

    class_ids   = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences >= min_conf_threshold
    if mask.any():
        boxes_f = boxes_raw[mask]
        confs_f = confidences[mask]
        ids_f   = class_ids[mask]

        # Scale from model input size to display frame size
        sx = imW / input_width
        sy = imH / input_height
        xmin = np.clip((boxes_f[:, 0] - boxes_f[:, 2] / 2) * sx, 0, imW).astype(int)
        ymin = np.clip((boxes_f[:, 1] - boxes_f[:, 3] / 2) * sy, 0, imH).astype(int)
        bw   = (boxes_f[:, 2] * sx).astype(int)
        bh   = (boxes_f[:, 3] * sy).astype(int)

        nms_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in zip(xmin, ymin, bw, bh)]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confs_f.tolist(), min_conf_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = nms_boxes[i]
                xmax, ymax = x + w, y + h
                cv2.rectangle(frame, (x, y), (xmax, ymax), (10, 255, 0), 2)

                label = f"{labels[ids_f[i]]}: {int(confs_f[i] * 100)}%"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ly = max(y, label_size[1] + 10)
                cv2.rectangle(frame, (x, ly - label_size[1] - 10), (x + label_size[0], ly + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x, ly - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('D20 Detector', frame)

    t2 = cv2.getTickCount()
    frame_rate_calc = freq / (t2 - t1)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
