"""Download pre-trained YOLO detection models into the models/ directory.

Usage:
    python scripts/download_models.py                 # yolov8n.pt
    python scripts/download_models.py -m yolov8s.pt   # medium-fast model
    python scripts/download_models.py -m yolov8n.pt -m yolo11n.pt

Nothing is trained; the models are official Ultralytics pre-trained COCO
checkpoints fetched from the ultralytics/assets GitHub release.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import BUILTIN_MODELS, download_file

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

_DESCRIPTIONS = {
    "yolov8n.pt": "YOLOv8n - fastest, smallest (~6.2 MB), great on CPU.",
    "yolov8s.pt": "YOLOv8s - good speed/accuracy balance (~22.5 MB).",
    "yolov8m.pt": "YOLOv8m - medium, higher accuracy (~49.7 MB).",
    "yolov8l.pt": "YOLOv8l - large, accurate (~83.7 MB).",
    "yolov8x.pt": "YOLOv8x - most accurate YOLOv8 (~130.5 MB).",
    "yolo11n.pt": "YOLO11n - newest nano model (~5.3 MB).",
    "yolo11s.pt": "YOLO11s - newest small model (~18.4 MB).",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model",
        action="append",
        default=None,
        choices=sorted(BUILTIN_MODELS),
        help="Model(s) to download. Repeat the flag for several models."
        " Default: yolov8n.pt.",
    )
    parser.add_argument(
        "-o", "--out",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
        help="Directory to save the weights into.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in args.model or ["yolov8n.pt"]:
        print(f"\n[{name}] {_DESCRIPTIONS.get(name, 'pre-trained COCO model')}")
        dest = os.path.join(args.out, name)
        download_file(f"{_ASSETS_BASE}/{name}", dest)
        print(f"Saved to {dest}")

    print("\nDone. Use a downloaded model with, e.g.:")
    print(f"  python src/main.py --source webcam --weights {os.path.join(args.out, args.model[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())