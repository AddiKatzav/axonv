"""
Shared contract for the Streamer → Detector → Displayer pipeline (Step A.1).

All components use these types and message formats so the pipeline stays consistent.
"""
from __future__ import annotations

from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Detection format (OpenCV-style bounding box)
# ---------------------------------------------------------------------------
# Each detection is (x, y, width, height) as returned by cv2.boundingRect().
# The Detector produces a list of these; the Displayer draws them (Step A; blur in Step B).
DetectionBox = Tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# End-of-stream sentinel
# ---------------------------------------------------------------------------
# When put on a queue, signals "no more frames". The receiver should forward
# it to the next stage (Detector → Displayer) and then exit.
SENTINEL: Any = None

# ---------------------------------------------------------------------------
# Message formats (for documentation; not enforced at runtime)
# ---------------------------------------------------------------------------
# Streamer → Detector:
#   Each item: (frame_index: int, frame: np.ndarray, fps: float)
#   frame: BGR image, shape (H, W, 3), dtype uint8
#   After last frame, Streamer puts SENTINEL.
#
# Detector → Displayer:
#   Each item: (frame_index: int, frame: np.ndarray, detections: list[DetectionBox], fps: float)
#   detections: list of (x, y, w, h); may be empty
#   After last frame, Detector forwards SENTINEL and exits.
#
# Displayer receives the above, draws/blurs, displays; on SENTINEL it exits.
