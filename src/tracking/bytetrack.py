"""ByteTrack tracker.

Implementation of the "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box" algorithm (Zhang et al., ECCV 2022).

Key idea
--------
Most trackers discard low-confidence detections outright, which breaks
tracks when an object is briefly occluded. ByteTrack instead splits every
object's detections into *high-score* (>= ``track_thresh``) and *low-score*
(< ``track_thresh``, >= ``min_conf``) parts and runs association in two
stages:

  1. high-score detections are matched to existing tracks with a strict
     IoU threshold (greedy Hungarian assignment);
  2. the low-score detections are matched to the *remaining* tracks with a
     looser IoU threshold, so temporarily-weak detections keep a track alive.

A standard SORT-style linear Kalman filter (constant velocity model over
``(x, y, aspect_ratio, height)``) predicts box positions between frames.

Fully self-contained except for numpy, so it can be swapped in/out of the
pipeline independently of any detection backend.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..detection.base import Detection
from .assignment import ious_xyxy, linear_assignment
from .base import TrackState, TrackedObject, Tracker


class KalmanBox:
    """Constant-velocity Kalman filter over ``[cx, cy, ar, h, vx, vy, va, vh]``.

    ``cx, cy`` is the box centre, ``ar`` the width/height aspect ratio and
    ``h`` the height. Measurements arrive as ``(cx, cy, ar, h)``. This is
    the filter formulation popularised by SORT.
    """

    _F = np.array(  # state transition (position + velocity, unit time step)
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    _H = np.array(  # measurement model: read position elements
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    _std_wp = 1.0 / 20.0    # standard-deviation weight for position
    _std_wv = 1.0 / 160.0   # standard-deviation weight for velocity

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None

    def initiate(self, xywh: Tuple[float, float, float, float]) -> None:
        """Initialise a track from a measurement box."""
        cx, cy, w, h = xywh
        ar = w / max(h, 1e-6)
        mean = np.array(
            [cx, cy, ar, h, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        std_pos = np.abs(mean[:4]) * self._std_wp
        std_vel = np.abs(mean[:4]) * self._std_wv
        # Element 2 (aspect ratio) is dimensionless, so its position variance
        # is tiny; velocities get a wider spread than positions.
        cov = np.diag(
            np.square(
                np.concatenate(
                    (
                        2.0 * std_pos[:2],
                        [1e-2],
                        [2.0 * std_pos[3]],
                        10.0 * std_vel[:2],
                        [1e-5],
                        [10.0 * std_vel[3]],
                    )
                )
            )
        )
        self.mean = mean
        self.cov = cov

    def predict(self) -> None:
        """Advance the state one time step (no control input)."""
        assert self.mean is not None and self.cov is not None

        std_pos = np.abs(self.mean[:4]) * self._std_wp
        std_vel = np.abs(self.mean[:4]) * self._std_wv
        q_pos = np.square(np.concatenate((std_pos[:2], [1e-2], [std_pos[3]])))
        q_vel = np.square(np.concatenate((std_vel[:2], [1e-5], [std_vel[3]])))
        q = np.diag(np.concatenate((q_pos, q_vel)))

        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + q

    def update(self, xywh: Tuple[float, float, float, float]) -> None:
        """Correct the prediction with a measured box."""
        assert self.mean is not None and self.cov is not None

        cx, cy, w, h = xywh
        ar = w / max(h, 1e-6)
        z = np.array([cx, cy, ar, h], dtype=np.float64)

        s = self._H @ self.cov @ self._H.T + np.eye(4)
        k = self.cov @ self._H.T @ np.linalg.inv(s)
        y = z - self._H @ self.mean
        self.mean = self.mean + k @ y
        self.cov = self.cov - k @ self._H @ self.cov

    def get_bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Decode the current state into an ``(x1, y1, x2, y2)`` box."""
        assert self.mean is not None
        cx, cy, ar, h = self.mean[:4]
        h = max(h, 1e-6)
        ar = max(ar, 1e-6)
        w = ar * h
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class STrack:
    """A single track: Kalman filter plus identity metadata."""

    def __init__(
        self,
        kalman: KalmanBox,
        track_id: int,
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.kalman = kalman
        self.track_id = track_id
        self.state = TrackState.NEW
        self.frames_since_update = 0
        self.hits = 1
        self.score = confidence
        self.class_id = class_id
        self.class_name = class_name

    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        return self.kalman.get_bbox_xyxy()

    def confirm(self) -> None:
        self.state = TrackState.TRACKED
        self.frames_since_update = 0

    def mark_lost(self) -> None:
        self.state = TrackState.LOST
        self.frames_since_update += 1

    def mark_matched(self, detection: Detection) -> None:
        self.kalman.update(detection.xywh)
        self.frames_since_update = 0
        self.hits += 1
        self.score = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.state = TrackState.TRACKED

    def remove(self) -> None:
        self.state = TrackState.REMOVED


class ByteTrack(Tracker):
    """ByteTrack multi-object tracker.

    Args:
        track_thresh: Detections with score >= this form the *high-score*
            set; they drive stage-1 association and may spawn new tracks.
        min_conf: Detections below this are discarded entirely. Detections in
            ``[min_conf, track_thresh)`` are the *low-score* set used in the
            second association stage to keep lost tracks alive.
        iou_thresh_high: Minimum IoU required for stage-1 matches.
        iou_thresh_low: Minimum IoU required for stage-2 matches.
        max_time_lost: Number of frames a track may stay lost before removal.
        min_hits: Frames a new track must be observed before it is confirmed
            and its ID is exposed to the rest of the pipeline.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        min_conf: float = 0.1,
        iou_thresh_high: float = 0.2,
        iou_thresh_low: float = 0.5,
        max_time_lost: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.track_thresh = float(track_thresh)
        self.min_conf = float(min_conf)
        self.iou_thresh_high = float(iou_thresh_high)
        self.iou_thresh_low = float(iou_thresh_low)
        self.max_time_lost = int(max_time_lost)
        self.min_hits = int(min_hits)

        self._tracks: List[STrack] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # state management
    # ------------------------------------------------------------------
    def _activate_track(self, detection: Detection) -> STrack:
        """Spawn, identity-assign and calibrate a new track."""
        kalman = KalmanBox()
        kalman.initiate(detection.xywh)
        track = STrack(
            kalman,
            track_id=self._next_id,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        self._next_id += 1
        track.confirm()
        return track

    def _cleanup(self) -> None:
        """Drop expired tracks from the pool."""
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

    # ------------------------------------------------------------------
    # main update
    # ------------------------------------------------------------------
    def update(self, detections: List[Detection], frame_shape: Tuple[int, int, int]) -> List[TrackedObject]:
        self._cleanup()

        if not detections:
            # No detections at all: every alive track goes one step lost.
            for track in self._tracks:
                if track.state == TrackState.TRACKED:
                    track.mark_lost()
                elif track.state == TrackState.LOST:
                    track.frames_since_update += 1
                    if track.frames_since_update > self.max_time_lost:
                        track.remove()
                elif track.state == TrackState.NEW:
                    track.mark_lost()
            self._cleanup()
            return self._output(frame_shape)

        # Split detections into the high-score and low-score sets.
        high_dets: List[Detection] = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets: List[Detection] = [
            d for d in detections if self.min_conf <= d.confidence < self.track_thresh
        ]

        # Candidate tracks: alive + recently-lost (bounded by max_time_lost).
        active: List[STrack] = [
            t for t in self._tracks
            if t.state in (TrackState.TRACKED, TrackState.NEW)
            or (t.state == TrackState.LOST and t.frames_since_update <= self.max_time_lost)
        ]
        if len(active) == 0 and not high_dets:
            return self._output(frame_shape)

        # Propagate every candidate track's state one step forward so the
        # box we match against is the *predicted* position, not the stale
        # last-measurement position. This also keeps the covariance alive.
        for track in active:
            track.kalman.predict()

        active_boxes = np.array([t.predicted_bbox() for t in active], dtype=np.float64)
        high_boxes = np.array([d.bbox_xyxy for d in high_dets], dtype=np.float64)
        low_boxes = np.array([d.bbox_xyxy for d in low_dets], dtype=np.float64)
        cost_high = 1.0 - ious_xyxy(high_boxes, active_boxes) if len(high_dets) else np.zeros((0, len(active)))
        cost_low = 1.0 - ious_xyxy(low_boxes, active_boxes) if len(low_dets) else np.zeros((0, len(active)))

        matched_high, unmatched_high, unmatched_tracks = linear_assignment(
            cost_high, threshold=1.0 - self.iou_thresh_high
        )
        for det_idx, track_idx in matched_high:
            active[track_idx].mark_matched(high_dets[det_idx])

        # Second stage: low-score detections rescue still-unmatched tracks.
        if len(unmatched_tracks) and len(low_dets):
            low_cost = cost_low[:, unmatched_tracks]
            matched_low, _, unmatched_tracks_2 = linear_assignment(
                low_cost, threshold=1.0 - self.iou_thresh_low
            )
            for det_idx, j in matched_low:
                active[unmatched_tracks[j]].mark_matched(low_dets[det_idx])
            unmatched_tracks = unmatched_tracks_2

        # Any track not matched this frame sinks one step towards being lost.
        for idx in unmatched_tracks:
            track = active[idx]
            if track.state == TrackState.TRACKED:
                track.mark_lost()
            elif track.state == TrackState.LOST:
                track.frames_since_update += 1
                if track.frames_since_update > self.max_time_lost:
                    track.remove()
            elif track.state == TrackState.NEW:
                track.mark_lost()

        # Unmatched high-score detections spawn new tracks.
        for det_idx in unmatched_high:
            if high_dets[det_idx].confidence >= self.track_thresh:
                self._tracks.append(self._activate_track(high_dets[det_idx]))

        self._cleanup()
        return self._output(frame_shape)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _output(self, frame_shape: Tuple[int, int, int]) -> List[TrackedObject]:
        """Project tracks back into boxes, clipped to the frame."""
        h, w = frame_shape[:2]
        out: List[TrackedObject] = []
        for track in self._tracks:
            if track.state not in (TrackState.TRACKED, TrackState.LOST, TrackState.NEW):
                continue
            if track.state == TrackState.NEW and track.hits < self.min_hits:
                continue
            x1, y1, x2, y2 = track.predicted_bbox()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            out.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=track.score,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    state=track.state,
                    frames_since_update=track.frames_since_update,
                )
            )
        return out