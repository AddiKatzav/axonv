"""
Streamer component (Step A.2): reads a video from an address (file path or URL),
extracts frame by frame, and sends each frame to the Detector via a queue.

Message format (see shared.py): (frame_index, frame, fps); then SENTINEL when done.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from shared import SENTINEL

logger = logging.getLogger(__name__)


def run_streamer(
    video_source: str,
    out_queue,
    *,
    max_frames: Optional[int] = None,
) -> None:
    """
    Read video frame by frame and push each to the detector queue.

    Args:
        video_source: Path or URL to the video (e.g. file.mp4 or rtsp://...).
        out_queue: multiprocessing.Queue; each item is (frame_index, frame, fps).
        max_frames: If set, stop after this many frames (for testing).

    Puts SENTINEL on out_queue when the video ends. Frame is BGR (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Streamer opened %s, fps=%.2f", video_source, fps)

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = np.ascontiguousarray(frame)
            out_queue.put((frame_index, frame, fps))
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()

    out_queue.put(SENTINEL)
    logger.info("Streamer finished after %d frames", frame_index)
