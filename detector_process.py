"""
Detector component: receives frames from the Streamer, runs motion detection,
and sends (frame, detections) to the Displayer. Does not draw on the image.
"""
from __future__ import annotations

import logging
from queue import Empty
from typing import Optional

import cv2
import numpy as np

try:
    import imutils
except ImportError:
    imutils = None

from shared import DetectionBox, PipelineStop, StopReason, get_stop_reason, is_stop

logger = logging.getLogger(__name__)


def _grab_contours(cnts):
    """
    Return contour list from cv2.findContours output (OpenCV 3/4 compatible).

    Args:
        cnts: Return value of cv2.findContours (tuple format varies by OpenCV version).

    Returns:
        List of contours (same as imutils.grab_contours when imutils is available).
    """
    if imutils is not None:
        return imutils.grab_contours(cnts)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def _contours_to_boxes(cnts, min_area: int) -> list[DetectionBox]:
    """
    Convert contours to bounding boxes, filtering by minimum area.

    Args:
        cnts: List of contours (e.g. from cv2.findContours).
        min_area: Minimum contour area; contours below this are dropped.

    Returns:
        List of DetectionBox (x, y, w, h) for contours with area >= min_area.
    """
    boxes: list[DetectionBox] = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes


def create_detector(min_area: int = 500):
    """
    Return a stateful detect(frame) for motion detection (closure over counter, prev_frame).

    Args:
        min_area: Minimum contour area to keep (small contours are filtered out).

    Returns:
        A callable detect(frame: np.ndarray) -> list[DetectionBox] (x, y, w, h per box).

    Algorithm: grayscale, absdiff, threshold(25), dilate(2), findContours; first frame
    returns [] and is used only as previous frame for the next diff.
    """
    counter = 0
    prev_frame = None

    def detect(frame: np.ndarray) -> list[DetectionBox]:
        # nonlocal so assignments here update create_detector's state across calls (closure).
        nonlocal counter, prev_frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == 0:
            # First frame: no previous frame to diff against, so no motion; store and return no detections.
            prev_frame = gray_frame
            counter += 1
            return []
        diff = cv2.absdiff(gray_frame, prev_frame)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = _grab_contours(cnts)
        prev_frame = gray_frame
        counter += 1
        return _contours_to_boxes(cnts, min_area)

    return detect


def run_detector(
    in_queue,
    out_queue,
    *,
    debug: bool = False,
    stop_requested: Optional[object] = None,
) -> None:
    """
    Detector process loop: read from in_queue, run motion detection, write to out_queue.

    Forwards PipelineStop unchanged so the stop reason is preserved. Exits on stop
    or when stop_requested is set (puts INTERRUPT then exits).

    Args:
        in_queue: Receives (frame_index, frame, fps) from Streamer; stop token to stop.
        out_queue: Sends (frame_index, frame, detections, fps) to Displayer; forwards stop.
        debug: If True, assert that the frame is unchanged after detect() (no drawing).
        stop_requested: Optional Event; if set, put PipelineStop(INTERRUPT) and exit.

    Returns:
        None.
    """
    detect = create_detector()
    count = 0
    poll_interval = 0.5
    try:
        while True:
            if stop_requested is not None and stop_requested.is_set():
                out_queue.put(PipelineStop(StopReason.INTERRUPT))
                logger.info("Detector exiting on stop_requested (INTERRUPT)")
                break
            try:
                item = in_queue.get(timeout=poll_interval)
            except Empty:
                continue
            if is_stop(item):
                out_queue.put(item)
                logger.info("Detector forwarding stop (reason=%s)", get_stop_reason(item).value)
                break
            frame_index, frame, fps = item
            if debug:
                frame_copy = frame.copy()
                detections = detect(frame)
                if not np.array_equal(frame, frame_copy):
                    logger.error(
                        "Detector modified the frame (frame_index=%s); "
                        "this component must not draw on the image.",
                        frame_index,
                    )
                    raise AssertionError(
                        "Detector must not modify the frame (draw on the image)"
                    )
                logger.debug("Frame %s unchanged by detector (OK)", frame_index)
            else:
                detections = detect(frame)
            out_queue.put((frame_index, frame, detections, fps))
            count += 1
    finally:
        logger.info("Detector processed %d frames", count)