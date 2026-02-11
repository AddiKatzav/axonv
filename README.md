# Video pipeline: Streamer → Detector → Displayer (Step A only)

Three-process pipeline for analytics on video streams: **Streamer** reads frames, **Detector** finds motion, **Displayer** draws detections and shows the video. This is the Step A implementation only (no blur, no Step B/C features).

## Step A.1 – Pipeline contract

All components share the same message format and sentinel (see `shared.py`):

- **DetectionBox:** `(x, y, width, height)` per OpenCV `boundingRect`.
- **Streamer → Detector:** `(frame_index, frame, fps)`; then `SENTINEL` when the video ends.
- **Detector → Displayer:** `(frame_index, frame, detections, fps)`; Detector forwards `SENTINEL` and exits.
- **SENTINEL:** Single value signalling end-of-stream; no more frames will be sent.

## Requirements

- Python 3.10+
- OpenCV (opencv-python), NumPy, imutils

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py /path/to/video.mp4
```

Optional: `--queue-size 3` (default) limits buffering between components for smoother playback. `--debug` enables a runtime check in the Detector that the frame is not modified (verifies the component does not draw on the image).

## Components

1. **Streamer** – Opens the video (file path or URL), reads frame by frame, sends `(frame_index, frame, fps)` to the Detector.
2. **Detector** – Runs OpenCV motion detection (background subtraction + contours). Sends `(frame_index, frame, detections, fps)` to the Displayer. **Does not draw** on the image.
3. **Displayer** – Draws rectangles for detections and the current video time (elapsed) in the top-left, and displays the frame at the video’s FPS.

When the video ends, the Streamer sends a sentinel; the Detector forwards it; the Displayer exits and closes the window.

## Inter-process communication (IPC)

**Choice: `multiprocessing.Queue`**

- **Why:** Simple, no extra dependencies, fits a linear pipeline (Streamer → Detector → Displayer). Each stage reads from one queue and writes to the next. Bounded `maxsize` gives back-pressure so the Streamer doesn’t run far ahead and playback stays in sync with the original video timing.
- **Alternatives considered:** Pipes (more low-level, single producer/consumer per pipe). ZeroMQ/sockets (extra dependency and complexity). Shared memory (more code for serialization and sync). For this assignment, `Queue` is sufficient and keeps the design clear.

