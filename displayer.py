"""
Displayer component (Step A): receives (frame, detections) from the Detector,
draws the detections and current time on the image, and displays the video.
Blur (Step B) is applied to each detection ROI using a NumPy-only box blur.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from shared import get_stop_reason, is_stop

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm.

    Args:
        seconds: Elapsed time in seconds (float).

    Returns:
        String like "00:01:23.456".
    """
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


def _box_blur_along_axis(grid: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """
    Box blur a 2D grid along one axis (vectorized). Shared implementation for
    row and column passes; uses cumsum and array indexing (no Python loop).

    Args:
        grid: 2D array (height, width). Any dtype; converted to float64.
        radius: Half-window size. Window length is 2*radius+1, or smaller at edges.
        axis: 0 = blur along columns (vertical), 1 = blur along rows (horizontal).

    Returns:
        Blurred 2D array, same shape, float64.
    """
    grid = np.asarray(grid, dtype=np.float64)
    length = grid.shape[axis]
    if length == 0:
        return grid
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (1, 0)
    padded = np.pad(grid, pad_width, constant_values=0)
    cumsum = np.cumsum(padded, axis=axis)
    index = np.arange(length)
    window_start = np.maximum(0, index - radius)
    window_end = np.minimum(length, index + radius + 1)
    if axis == 0:
        window_sum = cumsum[window_end, :] - cumsum[window_start, :]
        window_count = (window_end - window_start)[:, np.newaxis]
    else:
        window_sum = cumsum[:, window_end] - cumsum[:, window_start]
        window_count = window_end - window_start
    return window_sum / window_count


def _box_blur_along_rows(grid: np.ndarray, radius: int) -> np.ndarray:
    """Box blur along each row (horizontal pass). Delegates to _box_blur_along_axis(..., axis=1)."""
    return _box_blur_along_axis(grid, radius, axis=1)


def _box_blur_along_columns(grid: np.ndarray, radius: int) -> np.ndarray:
    """Box blur along each column (vertical pass). Delegates to _box_blur_along_axis(..., axis=0)."""
    return _box_blur_along_axis(grid, radius, axis=0)


def _clip_roi_to_image(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Clip detection box to image bounds and return ROI geometry.

    Args:
        img: Image with shape (H, W, ...).
        x, y, w, h: Top-left and size of the detection box.

    Returns:
        (roi_left, roi_top, roi_right, roi_bottom, roi_width, roi_height) if
        ROI has at least 2 pixels in each dimension; None otherwise.
    """
    img_height, img_width = img.shape[:2]
    roi_left = max(0, x)
    roi_top = max(0, y)
    roi_right = min(img_width, x + w)
    roi_bottom = min(img_height, y + h)
    roi_width = roi_right - roi_left
    roi_height = roi_bottom - roi_top
    if roi_width < 2 or roi_height < 2:
        return None
    return (roi_left, roi_top, roi_right, roi_bottom, roi_width, roi_height)


def _blur_roi(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    kernel_size: int = 15,
) -> None:
    """
    Blur the detection ROI in-place with a separable box blur (NumPy only).

    Args:
        img: BGR image, shape (H, W, 3), dtype uint8; modified in-place.
        x, y, w, h: Top-left and size of the ROI (detection box).
        kernel_size: Box kernel side length; forced to odd. Skipped or clamped for small ROIs.

    Returns:
        None. Modifies img in-place.
    """
    clipped = _clip_roi_to_image(img, x, y, w, h)
    if clipped is None:
        return
    roi_left, roi_top, roi_right, roi_bottom, roi_width, roi_height = clipped
    kernel_side = min(kernel_size, roi_width, roi_height) | 1
    if kernel_side < 2:
        return
    radius = kernel_side // 2

    roi = img[roi_top:roi_bottom, roi_left:roi_right]  # (roi_height, roi_width, 3)
    # Separable blur: horizontal pass then vertical pass, per channel (vectorized).
    for channel_idx in range(roi.shape[2]):
        channel = roi[:, :, channel_idx].astype(np.float64)
        channel = _box_blur_along_rows(channel, radius)
        channel = _box_blur_along_columns(channel, radius)
        roi[:, :, channel_idx] = np.clip(channel, 0, 255).astype(np.uint8)


def _draw_elapsed_time(
    img: np.ndarray,
    elapsed_seconds: float,
    x: int = 10,
    y: int = 30,
) -> None:
    """
    Draw video elapsed time on image (outline + fill for readability).

    Args:
        img: BGR image to draw on (modified in-place).
        elapsed_seconds: Time in seconds to display.
        x, y: Top-left position for the text (default 10, 30).

    Returns:
        None.
    """
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
    stop_requested: object | None = None,
) -> None:
    """
    Displayer process loop: read (frame_index, frame, detections, fps), blur ROIs, draw and show.

    Exits on stop message (logs reason) or when user closes the window (sets stop_requested
    so other processes can exit gracefully).

    Args:
        in_queue: Receives (frame_index, frame, detections, fps) from Detector; stop token to stop.
        window_name: Title of the OpenCV window.
        stop_requested: Optional Event; set when user closes window so pipeline can shut down.

    Returns:
        None.
    """
    first_ts: float | None = None
    frame_count = 0

    try:
        while True:
            item = in_queue.get()
            if is_stop(item):
                reason = get_stop_reason(item)
                logger.info("Displayer received stop (reason=%s), closing", reason.value)
                break

            frame_index, frame, detections, fps = item
            img = frame.copy()

            for (x, y, w, h) in detections:
                _blur_roi(img, x, y, w, h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elapsed = frame_index / fps if fps > 0 else 0.0
            _draw_elapsed_time(img, elapsed)

            cv2.imshow(window_name, img)
            frame_count += 1

            first_ts = frame_scheduler(first_ts, frame_count, fps)

            cv2.waitKey(1)  # process window events so display updates
            # User closed window: signal pipeline to stop gracefully
            if stop_requested is not None:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        stop_requested.set()
                        logger.info("Displayer: user closed window (USER_QUIT)")
                        break
                except cv2.error:
                    pass
    finally:
        cv2.destroyAllWindows()
        logger.info("Displayer showed %d frames", frame_count)
