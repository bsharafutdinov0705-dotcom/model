"""YOLO detector implementation."""

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO-based person detector."""

    def _load_model(self) -> None:
        """Load YOLO model.

        Note: If model file doesn't exist locally, ultralytics will
        automatically download it on first use.
        """
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects using YOLO and return annotated frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Annotated frame with detections drawn
        """
        kwargs = {
            "classes": self.classes if self.classes else None,
            "conf": self.conf_threshold,
        }
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz

        results = self.model(frame, **kwargs)[0]
        annotated = results.plot()
        return annotated
