import os
import numpy as np

from detectors.base import BaseDetector
from object_detection import ObjectDetector as _OD


class YOLOv8TFLiteDetector(BaseDetector):
    name = "yolov8_tflite"
    description = "YOLOv8n INT8 TFLite — production model (conf=0.3, nms=0.45)"

    def __init__(self):
        self._od = _OD(os.getenv("MODEL_FOLDER", "tflite1/custom_model_lite"))

    def detect(self, image: np.ndarray) -> list:
        return self._od.detect_array(image)
