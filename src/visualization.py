"""Frame drawing helpers for the detection/tracking overlay."""

from typing import Iterable, Optional, Tuple

import numpy as np

from .tracking.base import TrackedObject

_WINDOW = "Object Detection & Tracking"


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Stable, distinct BGR color derived from the tracking ID."""
    hue = (track_id * 61) % 360
    # HSV -> RGB, then BGR for OpenCV
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw_speed_badge(frame: np.ndarray, label: str) -> None:
    """Semi-transparent rounded badge in the top-left corner."""
    import cv2

    text = label
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    pad_x, pad_y = 10, 8
    x0, y0 = 10, 10
    x1, y1 = x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    cv2.putText(
        frame,
        text,
        (x0 + pad_x, y0 + pad_y + th - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def _draw_hint(frame: np.ndarray) -> None:
    import cv2

    text = "q: quit   |   space: pause"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = frame.shape[1] - tw - 12, frame.shape[0] - 12
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    tracks: Iterable[TrackedObject],
    fps: float,
    device: str = "",
    show_lost: bool = False,
) -> np.ndarray:
    """Overlay tracks, labels and the current FPS onto a copy of ``frame``."""
    import cv2

    out = frame.copy()
    for track in tracks:
        if track.is_lost and not show_lost:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]
        color = _color_for_id(track.track_id)
        thickness = 2 if not track.is_lost else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
        if track.is_lost:
            label += " (lost)"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = y1 - 6 if y1 - 6 > th + 4 else y2 + th + 6
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 6, label_y + baseline), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    fps_text = f"FPS: {fps:.1f}"
    if device:
        fps_text += f"  |  device: {device}"
    _draw_speed_badge(out, fps_text)
    _draw_hint(out)
    return out


def maybe_show(frame: np.ndarray, window_scale: Optional[float] = None) -> None:
    """Display the frame in a resizeable window (no-op if no display)."""
    import cv2

    if window_scale:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * window_scale), int(h * window_scale)))
    cv2.imshow(_WINDOW, frame)