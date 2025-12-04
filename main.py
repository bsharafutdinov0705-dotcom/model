#!/usr/bin/env python3
"""Main entry point for person detection comparison."""

import argparse
import sys
from pathlib import Path

from src.detectors import RTDETRDetector, YOLODetector
from src.utils.logger import logger


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare person detection models on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        choices=["yolo", "rtdetr"],
        help="Model to use for detection (yolo or rtdetr)",
    )
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output video file (default: input_name_model.mp4)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights file (default: yolo11x.pt or rtdetr-l.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold (default: 0.2)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size as height width (default: original video size)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.4,
        help="IoU threshold for RT-DETR NMS (default: 0.4)",
    )
    return parser


def main() -> int:
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()

    input_path = Path(args.input_video)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_{args.model}.mp4"

    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if args.model == "yolo":
            model_path = Path("yolo11x.pt")
        else:
            model_path = Path("rtdetr-l.pt")

    # Image size
    imgsz = None
    if args.imgsz:
        imgsz = tuple(args.imgsz)

    # Create detector
    try:
        if args.model == "yolo":
            detector = YOLODetector(
                model_path=model_path,
                conf_threshold=args.conf,
                classes=[0],  # Person class
                imgsz=imgsz,
            )
        else:  # rtdetr
            detector = RTDETRDetector(
                model_path=model_path,
                conf_threshold=args.conf,
                classes=[0],  # Person class
                imgsz=imgsz,
                iou_threshold=args.iou,
                agnostic_nms=True,
            )

        logger.info(f"Processing video with {args.model.upper()} model...")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Confidence threshold: {args.conf}")

        detector.process_video(input_path, output_path)

        logger.info(f"Success! Output saved to: {output_path}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info(
            "Note: Model weights will be downloaded automatically on first use."
        )
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
