"""Example variant: CLAHE contrast enhancement before detection.

Useful for low-light or low-contrast images. The L channel of LAB
colour space is equalised before passing to the model.
"""
import os
import cv2
import numpy as np

from detectors.base import BaseDetector
from object_detection import ObjectDetector as _OD


class CLAHEDetector(BaseDetector):
    name = "clahe_preprocess"
    description = "CLAHE contrast enhancement (LAB L-channel) before detection"

    def __init__(self):
        self._od = _OD(os.getenv("MODEL_FOLDER", "tflite1/custom_model_lite"))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, image: np.ndarray) -> list:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self._od.detect_array(enhanced)
