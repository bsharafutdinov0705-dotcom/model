"""Video processing utilities."""

from pathlib import Path

import cv2

from .logger import logger


def resize_video(
    input_path: str | Path,
    output_path: str | Path,
    new_width: int = 480,
    new_height: int = 270,
) -> None:
    """Resize video to new dimensions.

    Args:
        input_path: Path to input video file
        output_path: Path to save resized video file
        new_width: Target width in pixels
        new_height: Target height in pixels
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        msg = f"Error opening video file: {input_path}"
        raise ValueError(msg)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)
        frame_count += 1

    cap.release()
    out.release()

    logger.info(f"Resized {frame_count} frames. Output saved to {output_path}")
