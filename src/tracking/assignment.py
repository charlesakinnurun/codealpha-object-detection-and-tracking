"""Assignment (data association) helpers used by the trackers.

Contains a dependency-free O(n^3) Hungarian (Munkres) solver plus vectorised
IoU utilities. ``scipy`` is deliberately avoided so the tracker stays a pure
numpy dependency.
"""

from typing import List, Tuple

import numpy as np

_INF = 1e9


def iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two boxes in ``(x1, y1, x2, y2)`` format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def ious_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Broadcast IoU of every box in ``boxes_a`` against every box in ``boxes_b``.

    Args:
        boxes_a: ``(N, 4)`` array of ``(x1, y1, x2, y2)`` boxes.
        boxes_b: ``(M, 4)`` array of ``(x1, y1, x2, y2)`` boxes.

    Returns:
        ``(N, M)`` IoU matrix.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)
    return iou


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Minimum-cost assignment (Hungarian algorithm) on a rectangular matrix.

    Implements the classic O(min(n, m)^2 * max(n, m)) e-maxx formulation.

    Args:
        cost: ``(n, m)`` cost matrix (minimisation).

    Returns:
        List of ``(row, col)`` index pairs. Rows/columns without a partner
        are simply absent from the result.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    transposed = False
    if n == 0 or m == 0:
        return []
    if n > m:
        cost = cost.T
        n, m = m, n
        transposed = True

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)   # column -> row
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, _INF, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: List[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            pairs.append((p[j] - 1, j - 1))
    if transposed:
        pairs = [(c, r) for (r, c) in pairs]
    return pairs


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float = _INF,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve the assignment problem subject to a maximum-cost constraint.

    Args:
        cost_matrix: ``(n_dets, n_tracks)`` cost matrix (lower is better).
        threshold: Pairs whose cost exceeds the threshold are rejected.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` where matches is a list
        of ``(row, col)`` pairs, rows index the detections and cols the tracks.
    """
    if cost_matrix.size == 0:
        n_rows, n_cols = cost_matrix.shape
        return [], list(range(n_rows)), list(range(n_cols))

    pairs = hungarian(cost_matrix)
    matches: List[Tuple[int, int]] = []
    matched_rows = set()
    matched_cols = set()
    for (r, c) in pairs:
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    n_rows, n_cols = cost_matrix.shape
    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols