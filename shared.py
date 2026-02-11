"""
Shared contract for the Streamer → Detector → Displayer pipeline (Step A.1, Stage-C).

All components use these types and message formats so the pipeline stays consistent.
Explicit stop reasons ensure the video can stop for many reasons and all processes
exit gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Detection format (OpenCV-style bounding box)
# ---------------------------------------------------------------------------
# Each detection is (x, y, width, height) as returned by cv2.boundingRect().
# The Detector produces a list of these; the Displayer draws them (Step A; blur in Step B).
DetectionBox = Tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# Stop reasons (Stage-C): why the video / pipeline stopped
# ---------------------------------------------------------------------------
# Every process should exit gracefully; the reason is propagated so it can be logged.


class StopReason(str, Enum):
    """Reason the stream or pipeline stopped. Used for graceful shutdown and logging."""

    NORMAL_END = "normal_end"       # EOF or last frame (Streamer)
    OPEN_FAILED = "open_failed"    # VideoCapture failed to open (Streamer)
    READ_ERROR = "read_error"      # Frame read failed (Streamer)
    MAX_FRAMES = "max_frames"      # Reached max_frames limit, e.g. for testing (Streamer)
    USER_QUIT = "user_quit"       # User closed display window (Displayer)
    INTERRUPT = "interrupt"        # SIGINT/SIGTERM or main requested stop (Main)


@dataclass(frozen=True)
class PipelineStop:
    """
    End-of-stream message: no more frames. Receiver must forward to next stage and exit.

    Replaces the raw SENTINEL so the pipeline knows why the video stopped.
    """

    reason: StopReason


def is_stop(item: Any) -> bool:
    """True if item is a stop message (PipelineStop or legacy SENTINEL None)."""
    return item is None or isinstance(item, PipelineStop)


def get_stop_reason(item: Any) -> StopReason:
    """Return the stop reason; if item is legacy None, return NORMAL_END."""
    if isinstance(item, PipelineStop):
        return item.reason
    return StopReason.NORMAL_END


# Legacy: some code may still compare to SENTINEL. Prefer is_stop() and PipelineStop.
SENTINEL: Any = None

# ---------------------------------------------------------------------------
# Message formats (for documentation; not enforced at runtime)
# ---------------------------------------------------------------------------
# Streamer → Detector:
#   Each item: (frame_index: int, frame: np.ndarray, fps: float)
#   frame: BGR image, shape (H, W, 3), dtype uint8
#   After last frame or on error, Streamer puts PipelineStop(reason). Never raises
#   without putting a stop first (graceful shutdown).
#
# Detector → Displayer:
#   Each item: (frame_index: int, frame: np.ndarray, detections: list[DetectionBox], fps: float)
#   On receiving a stop (is_stop), Detector forwards the same PipelineStop and exits.
#
# Displayer: on stop, logs reason, closes window, exits. Optionally sets stop_requested
# when user closes the window so other processes can exit (USER_QUIT).
#
# Main: may set stop_requested Event on KeyboardInterrupt; processes check it and
# put PipelineStop(INTERRUPT) then exit.