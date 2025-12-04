"""RT-DETR detector implementation."""

from pathlib import Path

import numpy as np
from ultralytics import RTDETR

from .base import BaseDetector


class RTDETRDetector(BaseDetector):
    """RT-DETR-based person detector."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.2,
        classes: list[int] | None = None,
        imgsz: tuple[int, int] | None = None,
        iou_threshold: float = 0.4,
        agnostic_nms: bool = True,
    ) -> None:
        """Initialize RT-DETR detector.

        Args:
            model_path: Path to model weights file
            conf_threshold: Confidence threshold for detections
            classes: List of class IDs to detect (None for all classes)
            imgsz: Image size as (height, width) tuple
            iou_threshold: IoU threshold for NMS
            agnostic_nms: Whether to use agnostic NMS
        """
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        super().__init__(model_path, conf_threshold, classes, imgsz)

    def _load_model(self) -> None:
        """Load RT-DETR model.

        Note: If model file doesn't exist locally, ultralytics will
        automatically download it on first use.
        """
        self.model = RTDETR(str(self.model_path))

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects using RT-DETR and return annotated frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Annotated frame with detections drawn
        """
        kwargs = {
            "conf": self.conf_threshold,
            "classes": self.classes if self.classes else None,
            "iou": self.iou_threshold,
            "agnostic_nms": self.agnostic_nms,
        }
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz

        results = self.model(frame, **kwargs)[0]
        annotated = results.plot()
        return annotated
