"""Utility functions for video processing."""

from .logger import logger, setup_logger
from .video import resize_video

__all__ = ["resize_video", "logger", "setup_logger"]
