"""
Pipeline launcher (Step A only): runs Streamer, Detector, and Displayer as separate
processes, connected by bounded queues. Message formats and sentinel are defined
in shared.py.
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
    Returns 0 on success; non-zero on failure (e.g. missing video path,
    cannot open video, or pipeline/process error).
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

    # Queues use the contract in shared.py (Streamer→Detector, Detector→Displayer; SENTINEL to end)
    # Bounded maxsize: back-pressure so Streamer doesn't run ahead and playback stays in sync
    q_streamer_to_detector = mp.Queue(maxsize=args.queue_size)
    q_detector_to_displayer = mp.Queue(maxsize=args.queue_size)

    streamer = mp.Process(
        target=run_streamer,
        args=(args.video, q_streamer_to_detector),
        name="Streamer",
    )
    detector = mp.Process(
        target=run_detector,
        args=(q_streamer_to_detector, q_detector_to_displayer),
        kwargs={"debug": args.debug},
        name="Detector",
    )
    displayer = mp.Process(
        target=run_displayer,
        args=(q_detector_to_displayer,),
        kwargs={"window_name": "Pipeline output"},
        name="Displayer",
    )

    logger.info("Starting pipeline: %s", args.video)
    streamer.start()
    detector.start()
    displayer.start()

    streamer.join()
    detector.join()
    displayer.join()

    logger.info("Pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
