"""
Pipeline launcher (Step A + Stage-C): runs Streamer, Detector, and Displayer as separate
processes, connected by bounded queues. Message formats and stop contract are in shared.py.
Stage-C: explicit stop reasons (PipelineStop), graceful shutdown; all processes exit
when the video stops for any reason (EOF, error, user quit, interrupt).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys

from detector_process import run_detector
from displayer import run_displayer
from streamer import run_streamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Parse CLI (video path, --queue-size, --debug), create bounded queues and
    three processes (Streamer, Detector, Displayer), start and join them.
    On KeyboardInterrupt, requests graceful stop (all processes exit cleanly).
    Returns 0 on success; non-zero on failure (e.g. missing video path).
    """
    parser = argparse.ArgumentParser(
        description="Video pipeline: Streamer → Detector → Displayer (motion detection + display)",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="",
        help="Path or URL to the video (e.g. path/to/video.mp4)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=3,
        help="Max frames buffered between components (small = smoother playback, default 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detector runtime check: verify frame is not modified (no drawing on image)",
    )
    args = parser.parse_args()

    if not args.video:
        parser.error("video path or URL is required")
        return 1

    if args.debug:
        logging.getLogger("detector_process").setLevel(logging.DEBUG)

    # Queues and stop contract in shared.py; stop_requested allows graceful shutdown
    q_streamer_to_detector = mp.Queue(maxsize=args.queue_size)
    q_detector_to_displayer = mp.Queue(maxsize=args.queue_size)
    stop_requested = mp.Event()

    streamer = mp.Process(
        target=run_streamer,
        args=(args.video, q_streamer_to_detector),
        kwargs={"stop_requested": stop_requested},
        name="Streamer",
    )
    detector = mp.Process(
        target=run_detector,
        args=(q_streamer_to_detector, q_detector_to_displayer),
        kwargs={"debug": args.debug, "stop_requested": stop_requested},
        name="Detector",
    )
    displayer = mp.Process(
        target=run_displayer,
        args=(q_detector_to_displayer,),
        kwargs={"window_name": "Pipeline output", "stop_requested": stop_requested},
        name="Displayer",
    )

    logger.info("Starting pipeline: %s", args.video)
    streamer.start()
    detector.start()
    displayer.start()

    try:
        streamer.join(timeout=30)
        detector.join(timeout=10)
        displayer.join(timeout=10)
    except KeyboardInterrupt:
        logger.info("Interrupt requested; requesting graceful stop")
        stop_requested.set()
        streamer.join(timeout=5)
        detector.join(timeout=5)
        displayer.join(timeout=5)
        if streamer.is_alive() or detector.is_alive() or displayer.is_alive():
            logger.warning("One or more processes did not exit in time; terminating")
            streamer.terminate()
            detector.terminate()
            displayer.terminate()

    logger.info("Pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
