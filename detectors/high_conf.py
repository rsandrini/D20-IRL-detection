"""Example variant: same model, higher confidence threshold.

To add your own variant, copy this file, change `name`, `description`,
and tweak the threshold (or any other parameter).
"""
import os
import numpy as np

from detectors.base import BaseDetector
from object_detection import ObjectDetector as _OD


class HighConfDetector(BaseDetector):
    name = "high_conf"
    description = "Same model, stricter confidence threshold (conf=0.6, nms=0.45)"

    def __init__(self):
        self._od = _OD(os.getenv("MODEL_FOLDER", "tflite1/custom_model_lite"))
        self._od.min_conf_threshold = 0.6

    def detect(self, image: np.ndarray) -> list:
        return self._od.detect_array(image)
