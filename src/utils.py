"""Shared helpers: CLI source parsing, model download, video writer setup."""

import os
import subprocess
import sys
import urllib.request
from typing import Optional, Tuple

import numpy as np

# Names of pre-trained models we know how to download automatically.
BUILTIN_MODELS = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov5nu.pt", "yolov5su.pt",
}

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def parse_source(source: str):
    """Turn a CLI source into an OpenCV-capable input.

    Returns:
        ``int`` index for a webcam, or a filesystem path (str) for a video.
    """
    if source is None:
        return 0
    src = str(source).strip()
    if src.lower() == "webcam":
        return 0
    if src.isdigit():
        return int(src)
    return src


def download_file(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` with a progress bar showing bytes."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}")
    print(f"    -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        received = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            received += len(chunk)
            if total:
                pct = 100.0 * received / total
                print(f"\r    {received / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()


def ensure_model(weights: Optional[str]) -> str:
    """Return a usable model path, downloading a built-in one if needed."""
    weights = weights or "yolov8n.pt"
    name = os.path.basename(weights)

    if os.path.isfile(weights):
        return weights

    if name in BUILTIN_MODELS:
        url = f"{_ASSETS_BASE}/{name}"
        try:
            download_file(url, weights)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download pre-trained model '{name}'. "
                "Check your internet connection or place a valid .pt/.onnx "
                "model at the --weights path."
            ) from exc
        return weights

    raise FileNotFoundError(
        f"Model file '{weights}' does not exist and '{name}' is not a "
        "built-in pre-trained model. Use scripts/download_models.py to fetch "
        "one, or pass --weights pointing at an existing model file."
    )


def open_video_writer(
    path: str,
    fps: float,
    size: Tuple[int, int],
    fourcc: str = "mp4v",
) -> Optional["cv2.VideoWriter"]:
    """Create a ``cv2.VideoWriter`` with graceful fallbacks.

    Returns None if the fourcc codec cannot be created.
    """
    import cv2

    width, height = size
    if width <= 0 or height <= 0:
        width, height = 1920, 1080

    # User's choice first, then universally-available fallbacks.
    codecs = [fourcc, "mp4v", "XVID", "MJPG"]
    codecs = list(dict.fromkeys(codecs))
    for c in codecs:
        try:
            fourcc_code = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(path, fourcc_code, max(fps, 1.0), (width, height))
        except Exception:
            writer = None
        if writer is not None and writer.isOpened():
            return writer
        if writer is not None:
            writer.release()

    raise RuntimeError(
        f"Could not open video writer for '{path}' with any of codecs {codecs}. "
        "Try --fourcc avc1 (H.264) or opencv (use built-in Windows codecs)."
    )


def warn_missing_deps() -> None:
    """Report friendly messages when core third-party packages are absent."""
    reasons: list = []
    for module, pkg in (("cv2", "opencv-python"), ("ultralytics", "ultralytics")):
        try:
            __import__(module)
        except ImportError:
            reasons.append(pkg)
    if reasons:
        print(
            "Missing dependencies: " + ", ".join(reasons) + "\n"
            "Install them with:\n"
            "    pip install " + " ".join(reasons) + "\n",
            file=sys.stderr,
        )
        sys.exit(2)


def is_numeric(source) -> bool:
    return isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit())


def run_shell(command: str) -> None:
    """Run a shell command on any platform (informational helper)."""
    subprocess.run(command, shell=True, check=False)


def project_root() -> str:
    """Absolute path of the repository root (two levels above this file)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_array(frame) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame