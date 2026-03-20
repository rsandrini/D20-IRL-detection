from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list:
        """Run detection on a BGR numpy array.

        Returns list of dicts: {face: int, confidence: float, bbox: [x1, y1, x2, y2]}
        """
        ...
