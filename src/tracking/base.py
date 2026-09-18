"""Object tracking abstractions.

A ``Tracker`` consumes the per-frame output of a ``Detector`` and assigns
stable, unique tracking IDs to objects across frames. Different algorithms
(ByteTrack, SORT, Deep SORT, OC-SORT ...) can be plugged in behind this
interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

from ..detection.base import Detection


class TrackState(IntEnum):
    """Lifecycle state of a track."""

    NEW = 0          # spawned this frame, waiting to be confirmed
    TRACKED = 1      # confidently associated with detections
    LOST = 2         # temporarily without a matching detection
    REMOVED = 3      # expired, no longer usable


@dataclass
class TrackedObject:
    """A track as the rest of the application sees it."""

    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    state: TrackState = TrackState.TRACKED
    frames_since_update: int = field(default=0)

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST


class Tracker(ABC):
    """Contract every tracking implementation must fulfil."""

    @abstractmethod
    def update(self, detections: List[Detection], frame_shape: Tuple[int, int, int]) -> List[TrackedObject]:
        """Associate detections with existing tracks and update IDs.

        Args:
            detections: Detector output for the current frame (may be empty).
            frame_shape: ``(height, width, channels)`` of the current frame,
                used to keep predicted boxes inside the image.

        Returns:
            The list of active tracks for this frame.
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__