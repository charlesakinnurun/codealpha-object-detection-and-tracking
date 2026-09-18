"""Build a synthetic test clip with moving real-world objects.

Creates ``input/sample_tracking.mp4`` by sliding a viewing window over the
Ultralytics sample image ``bus.jpg`` (which contains people, a bus, cars,
etc.). The window moves at different directions over time, so tracked
objects translate on screen while keepers of their identity should be
preserved by the tracker — a simple offline verification for the whole
detect + track workflow without needing a camera or an external video.

Usage:
    python scripts/make_test_video.py
    python scripts/make_test_video.py --output input/sample_tracking.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

_SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_sample(dest: str) -> bool:
    if os.path.isfile(dest):
        return True
    from src.utils import download_file

    try:
        download_file(_SAMPLE_URL, dest)
        return True
    except Exception as exc:
        print(f"Could not fetch sample image {_SAMPLE_URL}: {exc}")
        return False


def render_sequence(img: np.ndarray, win: int = 640, seed: int = 7) -> None:
    """Yield a pan/zoom path over the image so objects move on screen."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    cx, cy = w / 2.0, h / 2.0
    target_cx, target_cy = w / 2.0, h / 2.0
    scale = 1.0
    for i, _ in enumerate(range(500)):
        if abs(target_cx - cx) < 3 and abs(target_cy - cy) < 3:
            target_cx = w * (0.25 + 0.5 * rng.random())
            target_cy = h * (0.25 + 0.5 * rng.random())
        cx += (target_cx - cx) * 0.08
        cy += (target_cy - cy) * 0.08

        scale = 1.0 - 0.25 * (0.5 + 0.5 * np.sin(i / 40.0))  # gentle slow zoom-in
        half = int(win * scale / 2)
        x1 = int(np.clip(cx - half, 0, max(w - win, 0)))
        y1 = int(np.clip(cy - half, 0, max(h - win, 0)))
        crop = img[y1 : y1 + win, x1 : x1 + win]
        frame = cv2.resize(crop, (win, win), interpolation=cv2.INTER_LINEAR)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (win, win), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
        cv2.putText(frame, "synthetic pan test - YOLO + ByteTrack", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        yield frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(_DIR, "input", "sample_tracking.mp4"),
        help="Path to write the generated video.",
    )
    parser.add_argument("--window", type=int, default=640, help="Crop window size in pixels.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second of the clip.")
    args = parser.parse_args()

    img_path = os.path.join(_DIR, "input", "bus.jpg")
    if not download_sample(img_path):
        return 1

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return 1

    window = min(args.window, img.shape[0], img.shape[1])
    if window != args.window:
        print(f"Note: window clamped to {window}px (image is only {img.shape[1]}x{img.shape[0]}).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (window, window))
    if not writer.isOpened():
        print(f"Could not create writer for {args.output}")
        return 1

    for frame in render_sequence(img, window):
        writer.write(frame)
    writer.release()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())