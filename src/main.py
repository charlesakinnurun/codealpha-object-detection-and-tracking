"""Task 4 - Object Detection and Tracking: command-line entry point.

Examples:
    python src/main.py --source webcam
    python src/main.py --source webcam --weights models/yolov8n.pt --conf 0.3
    python src/main.py --source input/video.mp4
    python src/main.py --source input/video.mp4 --output output/tracked.mp4
    python src/main.py --source 0 --no-display --output output/live.mp4
"""

import argparse
import os
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bootstrap_path() -> None:
    """Make ``src`` importable regardless of how this file is launched."""
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


_bootstrap_path()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="object-tracking",
        description="Real-time object detection and multi-object tracking (YOLO + ByteTrack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        default="webcam",
        help="Input: a webcam index (0, 1, ...), the keyword 'webcam', or a video file path.",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Pretrained detector weights (.pt/.onnx). Built-in models (e.g. yolov8n.pt) "
        "download automatically into the given location on first use.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated video (e.g. output/tracked.mp4).",
    )
    p.add_argument("--fourcc", default="mp4v", help="FourCC codec for video output.")
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Detections below it are used only to keep "
        "existing tracks alive; above it they create/update tracks and are drawn.",
    )
    p.add_argument(
        "--low-conf",
        type=float,
        default=0.1,
        help="Floor at which detections are emitted to the tracker at all.",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Detector inference resolution.")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU for the detector.")
    p.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto (GPU if available, else CPU), cpu, 0, cuda:0 ...",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Restrict detection to COCO class ids, e.g. --classes 0 (person) 2 (car).",
    )
    p.add_argument(
        "--tracker",
        choices=["bytetrack", "none"],
        default="bytetrack",
        help="Tracking algorithm. 'none' runs detection only.",
    )
    p.add_argument("--iou-track", type=float, default=0.2, help="ByteTrack stage-1 IoU threshold.")
    p.add_argument("--max-time-lost", type=int, default=30, help="Frames a track may stay lost before expiry.")
    p.add_argument(
        "--show-lost",
        action="store_true",
        help="Keep drawing tracks that currently have no detection.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a display window (useful for headless processing/saving).",
    )
    p.add_argument("--save-txt", action="store_true", help="Write per-frame tracking results to a CSV-like log.")
    p.add_argument(
        "--window-scale",
        type=float,
        default=0.9,
        help="Scale factor applied to the display window (1.0 = full size).",
    )
    return p


def _resolve_weights(args) -> str:
    """Pick weights: explicit --weights or a sensible default in models/."""
    if args.weights:
        return args.weights

    from src.utils import ensure_model

    # Default: keep the model inside the models/ directory.
    weights = os.path.join(_repo_root(), "models", "yolov8n.pt")
    return ensure_model(weights)


def _build_tracker(args, conf: float, low_conf: float):
    if args.tracker == "none":
        return None

    from src.tracking.bytetrack import ByteTrack

    return ByteTrack(
        track_thresh=conf,
        min_conf=low_conf,
        iou_thresh_high=args.iou_track,
        iou_thresh_low=0.5,
        max_time_lost=args.max_time_lost,
    )


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    from src.utils import parse_source, warn_missing_deps

    warn_missing_deps()
    src = parse_source(args.source)

    if isinstance(src, str) and not os.path.isfile(src):
        parser.error(f"Input video file not found: '{src}'. Place the clip in input/ or fix --source.")

    low_conf = max(0.0, min(args.low_conf, args.conf))
    if low_conf != args.low_conf:
        print(f"Note: --low-conf clamped to {low_conf:.2f} (must be <= --conf {args.conf:.2f}).")

    try:
        from src.detection.yolo_detector import YOLODetector
        from src.pipeline import TrackingPipeline

        weights = _resolve_weights(args)
        detector = YOLODetector(
            weights=weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=low_conf,
            iou=args.iou,
            classes=args.classes,
        )
        tracker = _build_tracker(args, conf=args.conf, low_conf=low_conf)

        pipeline = TrackingPipeline(
            source=src,
            detector=detector,
            tracker=tracker,
            output_path=args.output,
            fourcc=args.fourcc,
            display=not args.no_display,
            show_lost=args.show_lost,
            window_scale=args.window_scale,
            save_txt=args.save_txt,
        )
        pipeline.run()
        return 0
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())