"""Offline sanity check for the detection + tracking workflow.

Runs the YOLO detector and ByteTrack tracker over the first N frames of a
video source (not the full live loop), then reports:

  * device in use (GPU/CPU),
  * how many objects were detected per frame,
  * how many unique tracking IDs were assigned,
  * the single longest-lived track (ID continuity evidence).

It is a smoke test, not a benchmark: measured numbers depend on the machine.

Usage:
    python scripts/verify_workflow.py                        # default: input/sample_tracking.mp4
    python scripts/verify_workflow.py --source video.mp4 --frames 120
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector
from src.tracking.bytetrack import ByteTrack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.join("input", "sample_tracking.mp4"))
    parser.add_argument("--weights", default=os.path.join("models", "yolov8n.pt"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--conf", type=float, default=0.25, help="Display/new-track threshold.")
    parser.add_argument("--low-conf", type=float, default=0.1, help="Tracker emission floor.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: source '{args.source}' not found. Run scripts/make_test_video.py first.")
        return 1

    print("Initialising detector ...")
    detector = YOLODetector(weights=args.weights, device=args.device, conf=args.low_conf)
    tracker = ByteTrack(track_thresh=args.conf, min_conf=args.low_conf)
    detector.warmup()
    print(f"Device: {detector.device}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: could not open '{args.source}'.")
        return 1

    det_counts, track_counts, id_seen_frames = [], [], {}
    consumed = 0
    while consumed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = np.ascontiguousarray(frame)
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame.shape)

        det_counts.append(len(dets))
        track_counts.append(len([t for t in tracks if not t.is_lost]))
        for t in tracks:
            if not t.is_lost:
                id_seen_frames.setdefault(int(t.track_id), []).append(consumed)
        consumed += 1
    cap.release()

    if not det_counts:
        print("No frames processed - video appears empty.")
        return 1

    longest_id = max(id_seen_frames, key=lambda k: len(id_seen_frames[k])) if id_seen_frames else None
    longest_len = len(id_seen_frames[longest_id]) if longest_id is not None else 0

    print("\n--- workflow smoke test ---")
    print(f"source          : {args.source}")
    print(f"frames analysed : {consumed}")
    print(f"avg detections  : {np.mean(det_counts):.1f} / frame")
    print(f"avg tracks      : {np.mean(track_counts):.1f} / frame")
    print(f"unique track IDs: {len(id_seen_frames)}")
    print(f"longest-lived ID: #{longest_id} over {longest_len} frames")
    ok = longest_len >= max(4, consumed * 0.1) if consumed >= 2 else True
    print("result          :", "OK" if ok else "SUSPICIOUS - IDs churn very fast")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())