"""
Streamer component (Step A.2, Stage-C): reads a video from an address (file path or URL),
extracts frame by frame, and sends each frame to the Detector via a queue.

Always puts exactly one PipelineStop(reason) before exiting so the pipeline never blocks.
Message format (see shared.py): (frame_index, frame, fps); then PipelineStop(reason) when done.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from shared import PipelineStop, StopReason

logger = logging.getLogger(__name__)


def run_streamer(video_source: str, out_queue) -> None:
    """
    Read video frame by frame and push each to the detector queue.

    Puts exactly one PipelineStop(reason) on out_queue before exiting (normal end
    when video ends, open failed, or read error). Never leaves the pipeline
    waiting on an empty queue.

    Args:
        video_source: Path or URL to the video (e.g. file.mp4 or rtsp://...).
        out_queue: multiprocessing.Queue; each item is (frame_index, frame, fps).

    Returns:
        None.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        out_queue.put(PipelineStop(StopReason.OPEN_FAILED))
        logger.error("Cannot open video: %s", video_source)
        return

    stop_reason = StopReason.NORMAL_END
    frame_index = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info("Streamer opened %s, fps=%.2f", video_source, fps)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                stop_reason = StopReason.NORMAL_END
                break
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = np.ascontiguousarray(frame)
            out_queue.put((frame_index, frame, fps))
            frame_index += 1
    except Exception as e:
        logger.exception("Streamer read error: %s", e)
        stop_reason = StopReason.READ_ERROR
    finally:
        cap.release()
        out_queue.put(PipelineStop(stop_reason))
        logger.info("Streamer finished after %d frames (reason=%s)", frame_index, stop_reason.value)
