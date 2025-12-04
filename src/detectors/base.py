"""Base detector interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import logger


class BaseDetector(ABC):
    """Base class for object detectors."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.2,
        classes: list[int] | None = None,
        imgsz: tuple[int, int] | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            model_path: Path to model weights file
            conf_threshold: Confidence threshold for detections
            classes: List of class IDs to detect (None for all classes)
            imgsz: Image size as (height, width) tuple
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else []
        self.imgsz = imgsz
        self.model: Any = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model from model_path."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects in frame and return annotated frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Annotated frame with detections drawn
        """

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Process video file and save annotated output.

        Args:
            input_path: Path to input video file
            output_path: Path to save output video file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            msg = f"Error opening video file: {input_path}"
            raise ValueError(msg)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.detect(frame)
            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()

        logger.info(f"Processed {frame_count} frames. Output saved to {output_path}")
