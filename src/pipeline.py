"""Real-time detection-and-tracking pipeline.

Binds a ``Detector`` and a ``Tracker`` to an OpenCV video source (webcam
index or video file), overlays the results, optionally writes the processed
video to disk, and reports measured FPS.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .detection.base import Detector
from .tracking.base import TrackedObject, Tracker
from .utils import is_numeric, open_video_writer, parse_source
from .visualization import draw_frame, maybe_show

_PAUSE_KEY = ord(" ")
_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q and ESC


@dataclass
class PipelineStats:
    frames_processed: int = 0
    elapsed_sec: float = 0.0
    avg_fps: float = 0.0
    peak_fps: float = 0.0
    objects_tracked: int = 0
    max_track_id: int = 0
    writer_path: Optional[str] = None
    source_description: str = ""
    errors: list = field(default_factory=list)


class TrackingPipeline:
    """Run detection + tracking continuously over a video source."""

    def __init__(
        self,
        source,
        detector: Detector,
        tracker: Optional[Tracker],
        output_path: Optional[str] = None,
        fourcc: str = "mp4v",
        display: bool = True,
        show_lost: bool = False,
        window_scale: Optional[float] = 0.9,
        save_txt: bool = False,
    ) -> None:
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.output_path = output_path
        self.fourcc = fourcc
        self.display = display
        self.show_lost = show_lost
        self.window_scale = window_scale
        self.save_txt = save_txt

        self._writer = None
        self._cap = None
        self._txt_log = None
        self._source_desc = ""
        self.stats = PipelineStats()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _describe_source(self) -> str:
        if is_numeric(self.source):
            return f"webcam (index {self.source})"
        return f"video file: {self.source}"

    def _open_capture(self):
        import cv2

        src = parse_source(self.source)
        self._source_desc = self._describe_source()
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open {self._source_desc}. "
                "Check the device/index and that the video format is supported "
                "(MP4, AVI, MOV, MKV, etc.) by OpenCV."
            )

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not is_numeric(self.source) and fps <= 0.0:
            fps = 25.0
        return fps

    def _open_writer(self, fps: float, size: Tuple[int, int]):
        if not self.output_path:
            return None
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        return open_video_writer(self.output_path, fps, size, self.fourcc)

    def _open_txt_log(self):
        if not self.save_txt:
            return None
        if not self.output_path:
            return None
        txt_path = os.path.splitext(self.output_path)[0] + "_tracks.txt"
        return open(txt_path, "w")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> PipelineStats:
        import cv2

        try:
            # Open the source first so a dead camera fails fast, before the
            # (potentially slow) first-inference warmup.
            source_fps = self._open_capture()
            self.detector.warmup()
            print(f"Running on {self._source_desc} -> device: {self.detector.device}")
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._writer = self._open_writer(source_fps, (w, h))
            if self.output_path:
                self.stats.writer_path = self.output_path
            self._txt_log = self._open_txt_log()

            return self._loop(cv2, source_fps)
        finally:
            if self._cap is not None:
                self._cap.release()
            if self._writer is not None:
                self._writer.release()
            if self._txt_log is not None:
                self._txt_log.close()
            cv2.destroyAllWindows()

    def _loop(self, cv2, source_fps: float) -> PipelineStats:
        start = time.perf_counter()
        ema_fps = 0.0
        empty_reads = 0
        paused = False

        while True:
            if paused:
                key = cv2.waitKey(0) & 0xFF
                if key == _PAUSE_KEY:
                    paused = False
                elif key in _QUIT_KEYS:
                    break
                continue

            tick = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                empty_reads += 1
                # Webcams sometimes skip a frame; only give up after many.
                if empty_reads >= (30 if is_numeric(self.source) else 1):
                    break
                continue
            empty_reads = 0
            frame = np.ascontiguousarray(frame)

            detections = self.detector.detect(frame)

            if self.tracker is not None:
                tracks = self.tracker.update(detections, frame.shape)
            else:
                tracks = [
                    TrackedObject(
                        track_id=i + 1,
                        bbox_xyxy=d.bbox_xyxy,
                        confidence=d.confidence,
                        class_id=d.class_id,
                        class_name=d.class_name,
                    )
                    for i, d in enumerate(detections)
                ]

            # FPS (exponential moving average)
            dt = max(time.perf_counter() - tick, 1e-6)
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps == 0.0 else ema_fps * 0.9 + inst_fps * 0.1
            self.stats.peak_fps = max(self.stats.peak_fps, inst_fps)

            annotated = draw_frame(frame, tracks, ema_fps, self.detector.device, self.show_lost)

            if self._writer is not None:
                self._writer.write(annotated)
            if self._txt_log is not None:
                self._write_txt_log(tracks)

            self._update_stats(tracks, ema_fps)
            if self.display:
                maybe_show(annotated, self.window_scale)
                key = cv2.waitKey(1) & 0xFF
                if key == _PAUSE_KEY:
                    paused = True
                if key in _QUIT_KEYS:
                    break

        self._finalize_stats(start)
        return self.stats

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _write_txt_log(self, tracks) -> None:
        frame_idx = self.stats.frames_processed
        for t in tracks:
            if t.is_lost:
                continue
            x1, y1, x2, y2 = [f"{float(v):.2f}" for v in t.bbox_xyxy]
            self._txt_log.write(
                f"{frame_idx},{t.track_id},{t.class_name},{t.confidence:.3f},{x1},{y1},{x2},{y2}\n"
            )

    def _update_stats(self, tracks, ema_fps: float) -> None:
        self.stats.frames_processed += 1
        active = [t for t in tracks if not t.is_lost]
        self.stats.objects_tracked += len(active)
        if tracks:
            self.stats.max_track_id = max(int(t.track_id) for t in tracks)

    def _finalize_stats(self, wall_start: float) -> None:
        self.stats.elapsed_sec = time.perf_counter() - wall_start
        frames = self.stats.frames_processed
        self.stats.avg_fps = frames / self.stats.elapsed_sec if self.stats.elapsed_sec > 0 else 0.0

        print("\n--- session summary ---")
        print(f"source         : {self._source_desc}")
        print(f"frames read    : {frames}")
        print(f"elapsed        : {self.stats.elapsed_sec:.2f} s")
        print(f"avg FPS        : {self.stats.avg_fps:.2f}")
        print(f"peak FPS       : {self.stats.peak_fps:.2f}")
        print(f"objects seen   : {self.stats.objects_tracked}")
        print(f"highest track #: {self.stats.max_track_id}")
        if self.stats.writer_path:
            print(f"video saved to : {self.stats.writer_path}")
        if self._txt_log is not None:
            print("track log saved alongside the video output")