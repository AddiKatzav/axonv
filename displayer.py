"""
Displayer component (Step A): receives (frame, detections) from the Detector,
draws the detections and current time on the image, and displays the video.
Only this component draws on the image. No blur (Step B).
"""
from __future__ import annotations

import logging
import time

import cv2

from shared import SENTINEL

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def frame_scheduler(
    first_ts: float | None,
    frame_count: int,
    fps: float,
) -> float:
    """
    Throttle display to match the video frame rate (sleep until target show time).

    Args:
        first_ts: Start timestamp (None on first call; then set from perf_counter).
        frame_count: Number of frames shown so far (1-based for scheduling).
        fps: Frames per second; target time is first_ts + frame_count / fps.

    Returns:
        The start timestamp (same as first_ts after first call) for the next call.
    """
    if first_ts is None:
        first_ts = time.perf_counter()
    target_ts = first_ts + frame_count / fps
    now = time.perf_counter()
    sleep_time = target_ts - now
    if sleep_time > 0:
        time.sleep(sleep_time)
    return first_ts


def _draw_elapsed_time(img, elapsed_seconds: float, x: int = 10, y: int = 30) -> None:
    """Draw video elapsed time on image (outline + fill for readability)."""
    time_str = _format_time(elapsed_seconds)
    cv2.putText(
        img, time_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, time_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 0, 0), 1, cv2.LINE_AA,
    )


def run_displayer(
    in_queue,
    *,
    window_name: str = "Pipeline output",
) -> None:
    """
    Displayer process loop: read (frame_index, frame, detections, fps), draw and show.

    Args:
        in_queue: Receives (frame_index, frame, detections, fps) from Detector; SENTINEL to stop.
        window_name: Title of the OpenCV window.

    Draws rectangles for detections and elapsed time in the top-left; throttles to video FPS.
    Exits on SENTINEL and closes the window.
    """
    first_ts: float | None = None
    frame_count = 0

    try:
        while True:
            item = in_queue.get()
            if item is SENTINEL:
                break

            frame_index, frame, detections, fps = item
            img = frame.copy()

            for (x, y, w, h) in detections:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elapsed = frame_index / fps if fps > 0 else 0.0
            _draw_elapsed_time(img, elapsed)

            cv2.imshow(window_name, img)
            frame_count += 1

            first_ts = frame_scheduler(first_ts, frame_count, fps)

            cv2.waitKey(1)  # process window events so display updates
    finally:
        cv2.destroyAllWindows()
        logger.info("Displayer showed %d frames", frame_count)
