"""Machine Learning - Object Tracking - OCSORT Association Utilities

This module provides functions for associating detections with existing trackers
using various metrics such as Intersection over Union (IoU), Generalized IoU (GIoU),
Distance-IoU (DIoU), Complete-IoU (CIoU), and center distance. It also includes
functions for solving the linear assignment problem using the Hungarian algorithm.


Functions:
    - compute_iou_matrix: Computes the IoU matrix between two sets of bounding boxes.
    - compute_giou_matrix: Computes the GIoU matrix between two sets of bounding boxes.
    - compute_diou_matrix: Computes the DIoU matrix between two sets of bounding boxes.
    - compute_ciou_matrix: Computes the CIoU matrix between two sets of bounding boxes.
    - compute_dist_matrix: Computes the center distance matrix between two sets of bounding boxes.
    - compute_vdc_pairwise: Computes normalized velocity direction vectors for VDC.
    - solve_linear_assignment: Solves the linear assignment problem using Hungarian algorithm.
    - associate: Associates current detections to existing trackers using IoU and VDC.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

try:
    import lap

    has_lap: bool = True
except ImportError:
    has_lap: bool = False
    try:
        from scipy.optimize import linear_sum_assignment

        has_scipy: bool = True
    except ImportError:
        has_scipy: bool = False


def compute_iou_matrix(a_dets: npt.NDArray[np.floating], a_preds: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        a_dets (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (N, 4+).
        a_preds (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (M, 4+).

    Returns:
        npt.NDArray[np.floating]:
            IoU matrix of shape (N, M) where element [i,j] is IoU between a_dets[i] and a_preds[j].

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        This function uses the formula for IoU: Area of Intersection / Area of Union.
    """
    a_preds = np.expand_dims(a_preds, 0)
    a_dets = np.expand_dims(a_dets, 1)
    xx1 = np.maximum(a_dets[..., 0], a_preds[..., 0])
    yy1 = np.maximum(a_dets[..., 1], a_preds[..., 1])
    xx2 = np.minimum(a_dets[..., 2], a_preds[..., 2])
    yy2 = np.minimum(a_dets[..., 3], a_preds[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (a_dets[..., 2] - a_dets[..., 0]) * (a_dets[..., 3] - a_dets[..., 1])
        + (a_preds[..., 2] - a_preds[..., 0]) * (a_preds[..., 3] - a_preds[..., 1])
        - wh
    )
    return o


def compute_giou_matrix(
    a_dets: npt.NDArray[np.floating], a_preds: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Compute the Generalized Intersection over Union (GIoU) between two sets of bounding boxes.

    Args:
        a_dets (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (N, 4+).
        a_preds (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (M, 4+).

    Returns:
        npt.NDArray[np.floating]:
            GIoU matrix of shape (N, M) where element [i,j] is GIoU between a_dets[i] and a_preds[j].

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on GIoU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    a_preds = np.expand_dims(a_preds, 0)
    a_dets = np.expand_dims(a_dets, 1)
    xx1 = np.maximum(a_dets[..., 0], a_preds[..., 0])
    yy1 = np.maximum(a_dets[..., 1], a_preds[..., 1])
    xx2 = np.minimum(a_dets[..., 2], a_preds[..., 2])
    yy2 = np.minimum(a_dets[..., 3], a_preds[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (a_dets[..., 2] - a_dets[..., 0]) * (a_dets[..., 3] - a_dets[..., 1])
        + (a_preds[..., 2] - a_preds[..., 0]) * (a_preds[..., 3] - a_preds[..., 1])
        - wh
    )
    iou = wh / union
    xxc1 = np.minimum(a_dets[..., 0], a_preds[..., 0])
    yyc1 = np.minimum(a_dets[..., 1], a_preds[..., 1])
    xxc2 = np.maximum(a_dets[..., 2], a_preds[..., 2])
    yyc2 = np.maximum(a_dets[..., 3], a_preds[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.0) / 2.0
    return giou


def compute_diou_matrix(
    a_dets: npt.NDArray[np.floating], a_preds: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Compute the Distance-IoU (DIoU) between two sets of bounding boxes.

    Args:
        a_dets (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (N, 4+).
        a_preds (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (M, 4+).

    Returns:
        npt.NDArray[np.floating]:
            DIoU matrix of shape (N, M) where element [i,j] is DIoU between a_dets[i] and a_preds[j].

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on DIoU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    a_preds = np.expand_dims(a_preds, 0)
    a_dets = np.expand_dims(a_dets, 1)
    xx1 = np.maximum(a_dets[..., 0], a_preds[..., 0])
    yy1 = np.maximum(a_dets[..., 1], a_preds[..., 1])
    xx2 = np.minimum(a_dets[..., 2], a_preds[..., 2])
    yy2 = np.minimum(a_dets[..., 3], a_preds[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (a_dets[..., 2] - a_dets[..., 0]) * (a_dets[..., 3] - a_dets[..., 1])
        + (a_preds[..., 2] - a_preds[..., 0]) * (a_preds[..., 3] - a_preds[..., 1])
        - wh
    )
    iou = wh / union
    centerx1 = (a_dets[..., 0] + a_dets[..., 2]) / 2.0
    centery1 = (a_dets[..., 1] + a_dets[..., 3]) / 2.0
    centerx2 = (a_preds[..., 0] + a_preds[..., 2]) / 2.0
    centery2 = (a_preds[..., 1] + a_preds[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    xxc1 = np.minimum(a_dets[..., 0], a_preds[..., 0])
    yyc1 = np.minimum(a_dets[..., 1], a_preds[..., 1])
    xxc2 = np.maximum(a_dets[..., 2], a_preds[..., 2])
    yyc2 = np.maximum(a_dets[..., 3], a_preds[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag
    return (diou + 1) / 2.0


def compute_ciou_matrix(
    a_dets: npt.NDArray[np.floating], a_preds: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Compute the Complete-IoU (CIoU) between two sets of bounding boxes.

    Args:
        a_dets (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (N, 4+).
        a_preds (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (M, 4+).

    Returns:
        npt.NDArray[np.floating]:
            CIoU matrix of shape (N, M) where element [i,j] is CIoU between a_dets[i] and a_preds[j].

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on CIoU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    a_preds = np.expand_dims(a_preds, 0)
    a_dets = np.expand_dims(a_dets, 1)
    xx1 = np.maximum(a_dets[..., 0], a_preds[..., 0])
    yy1 = np.maximum(a_dets[..., 1], a_preds[..., 1])
    xx2 = np.minimum(a_dets[..., 2], a_preds[..., 2])
    yy2 = np.minimum(a_dets[..., 3], a_preds[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (a_dets[..., 2] - a_dets[..., 0]) * (a_dets[..., 3] - a_dets[..., 1])
        + (a_preds[..., 2] - a_preds[..., 0]) * (a_preds[..., 3] - a_preds[..., 1])
        - wh
    )
    iou = wh / union
    centerx1 = (a_dets[..., 0] + a_dets[..., 2]) / 2.0
    centery1 = (a_dets[..., 1] + a_dets[..., 3]) / 2.0
    centerx2 = (a_preds[..., 0] + a_preds[..., 2]) / 2.0
    centery2 = (a_preds[..., 1] + a_preds[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    xxc1 = np.minimum(a_dets[..., 0], a_preds[..., 0])
    yyc1 = np.minimum(a_dets[..., 1], a_preds[..., 1])
    xxc2 = np.maximum(a_dets[..., 2], a_preds[..., 2])
    yyc2 = np.maximum(a_dets[..., 3], a_preds[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    w1 = a_dets[..., 2] - a_dets[..., 0]
    h1 = a_dets[..., 3] - a_dets[..., 1]
    w2 = a_preds[..., 2] - a_preds[..., 0]
    h2 = a_preds[..., 3] - a_preds[..., 1]
    h2 = h2 + 1.0
    h1 = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    return (ciou + 1) / 2.0


def compute_dist_matrix(
    a_dets: npt.NDArray[np.floating], a_preds: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Measure the center distance between two sets of bounding boxes.

    Args:
        a_dets (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (N, 4+).
        a_preds (npt.NDArray[np.floating]): Bounding boxes in the format [x1, y1, x2, y2] and shape (M, 4+).

    Returns:
            Distance matrix of shape (N, M) where element [i,j] is the normalized distance
            between the center of a_dets[i] and a_preds[j].

    Note:
        This is a coarse implementation, and it may not be suitable for precise association.
        The returned values are normalized to the range (0, 1).
    """
    a_preds = np.expand_dims(a_preds, 0)
    a_dets = np.expand_dims(a_dets, 1)
    centerx1 = (a_dets[..., 0] + a_dets[..., 2]) / 2.0
    centery1 = (a_dets[..., 1] + a_dets[..., 3]) / 2.0
    centerx2 = (a_preds[..., 0] + a_preds[..., 2]) / 2.0
    centery2 = (a_preds[..., 1] + a_preds[..., 3]) / 2.0
    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2
    ct_dist = np.sqrt(ct_dist2)
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist


def compute_vdc_pairwise(
    a_dets: npt.NDArray[np.floating], a_trks: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute normalized velocity direction vectors between detections and tracks.

    Calculates unit direction vectors from each track center to each detection center,
    used for Velocity Direction Consistency (VDC) in object tracking association.

    Args:
        dets (npt.NDArray[np.floating]):
            Detection bounding boxes with shape (N, 4+).
            Format: [x1, y1, x2, y2, ...] where (x1,y1) is top-left and (x2,y2) is bottom-right.
            Additional columns (like confidence scores) are ignored.

        tracks (npt.NDArray[np.floating]):
            Track bounding boxes with shape (M, 4+).
            Format: [x1, y1, x2, y2, ...] representing previous track positions.
            Additional columns are ignored.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:
                - dy (numpy.ndarray): Y-direction components with shape (M, N).
                Normalized values in range [-1, 1].
                - dx (numpy.ndarray): X-direction components with shape (M, N).
                Normalized values in range [-1, 1].

            Where M is the number of tracks and N is the number of detections.
            Each element [i,j] represents the normalized direction vector from track[i] center
            to detection[j] center.

    Note:
        - Direction vectors are normalized to unit length (magnitude = 1)
        - Small epsilon (1e-6) is added to avoid division by zero
        - Used in VDC (Velocity Direction Consistency) for tracking association
        - Centers are computed as: center = [(x1+x2)/2, (y1+y2)/2]
    """
    a_trks = a_trks[..., np.newaxis]
    dets_cx, dets_cy = (a_dets[:, 0] + a_dets[:, 2]) / 2.0, (a_dets[:, 1] + a_dets[:, 3]) / 2.0
    trks_cx, trks_cy = (a_trks[:, 0] + a_trks[:, 2]) / 2.0, (a_trks[:, 1] + a_trks[:, 3]) / 2.0
    dx = dets_cx - trks_cx
    dy = dets_cy - trks_cy
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def solve_linear_assignment(a_cost_matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
    """
    Solve the linear assignment problem using Hungarian algorithm.

    Finds the optimal one-to-one assignment between detections and trackers
    that minimizes the total cost. Uses lap library if available (preferred
    for rectangular matrices), otherwise falls back to scipy.

    Args:
        a_cost_matrix (npt.NDArray[np.floating]):
            Cost matrix with shape (M, N) where a_cost_matrix[i,j] represents the cost of assigning
            detection i to tracker j. M is number of detections, N is number of trackers.

    Returns:
        npt.NDArray[np.integer]:
            Array of optimal assignments with shape (K, 2) where each row [i, j] represents
            detection i assigned to tracker j. K is the number of successful assignments (≤ min(M, N)).

    Raises:
        ImportError: If neither lap nor scipy is available for assignment solving.

    Note:
        - The lap library is preferred as it handles rectangular cost matrices better
        - For square matrices, both lap and scipy give identical results
        - Returns empty array if a_cost_matrix is empty
    """
    if a_cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    if has_lap:
        _, x, y = lap.lapjv(a_cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0], dtype=int)
    if has_scipy:
        row_indices, col_indices = linear_sum_assignment(a_cost_matrix)
        return np.column_stack([row_indices, col_indices])
    raise ImportError(
        "Neither 'lap' nor 'scipy' is available for linear assignment. "
        "Please install one of them: 'pip install lap' or 'pip install scipy'"
    )


def associate(
    a_dets: npt.NDArray[np.float32],
    a_trks: npt.NDArray[np.float32],
    a_iou_thre: float,
    a_vels: npt.NDArray[np.float32],
    a_prev_obs: npt.NDArray[np.float32],
    a_vdc_weight: float,
) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Associate current detections to existing trackers using IoU and Velocity Direction Consistency.

    This is the core association function in OC-SORT that combines geometric similarity (IoU)
    with motion consistency (VDC) to create robust detection-tracker associations.

    Algorithm:
    1. Compute velocity direction vectors between detections and previous observations
    2. Calculate angle differences between predicted velocities and observed directions
    3. Combine IoU similarity matrix with VDC cost matrix
    4. Solve assignment problem using Hungarian algorithm
    5. Filter matches below IoU threshold

    Args:
        a_dets (npt.NDArray[np.float32]):
            Current detection bounding boxes with shape (N, 5+).
            Format: [x1, y1, x2, y2, confidence, ...] where:
                - (x1, y1): top-left corner coordinates
                - (x2, y2): bottom-right corner coordinates
                - confidence: detection confidence score [0, 1]
                - Additional columns ignored

        a_trks (npt.NDArray[np.float32]):
            Current tracker predicted bounding boxes with shape (M, 4+).
            Format: [x1, y1, x2, y2, ...] representing predicted positions. Additional columns ignored.

        a_iou_thre (float): IoU threshold for valid matches, typically in range [0.1, 0.5].
            Matches with IoU below this value are rejected.

        a_vels (npt.NDArray[np.float32]):
            Tracker velocity vectors with shape (M, 2). Format: [vel_x, vel_y] representing normalized velocity
            direction for each tracker. Values typically in range [-1, 1] as unit vectors.

        a_prev_obs (npt.NDArray[np.float32]):
            Previous track observations with shape (M, 5+). Format: [x1, y1, x2, y2, score, label, ...] where:
                - score: track score, negative values indicate invalid tracks
                - Used to compute direction vectors for VDC

        a_vdc_weight (float):
            Weight for velocity direction consistency cost, typically in range [0, 1]. Higher values prioritize motion
            consistency over geometric similarity.

    Returns:
        Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
            A tuple containing:
                - matches (npt.NDArray[np.integer]):
                    Valid matches with shape (K, 2) where K ≤ min(N, M). Format: [detection_idx, tracker_idx] for each
                    successful match.
                - unmatched_detections (npt.NDArray[np.integer]):
                    Indices of unmatched detections with shape (U,). These detections may spawn new tracks.
                - unmatched_trackers (npt.NDArray[np.integer]):
                    Indices of unmatched trackers with shape (V,). These trackers may be marked for deletion.

            Where: K + U = N (total detections) and K + V = M (total trackers)

    Raises:
        ValueError: If array dimensions are incompatible or thresholds are invalid.

    Note:
        - Function handles empty inputs gracefully
        - VDC weight of 0 disables motion consistency (pure IoU matching)
        - Higher IoU thresholds result in more conservative matching
        - Invalid previous observations (age < 0) are masked out from VDC computation
    """
    if len(a_trks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(a_dets)),
            np.empty((0, 5), dtype=int),
        )

    dy_curr, dx_curr = compute_vdc_pairwise(a_dets, a_prev_obs)
    dy_pred, dx_pred = a_vels[:, 0], a_vels[:, 1]
    dy_pred = np.repeat(dy_pred[:, np.newaxis], dy_curr.shape[1], axis=1)
    dx_pred = np.repeat(dx_pred[:, np.newaxis], dx_curr.shape[1], axis=1)
    diff_angle_cos = dx_pred * dx_curr + dy_pred * dy_curr
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(a_prev_obs.shape[0])
    valid_mask[np.where(a_prev_obs[:, 4] < 0)] = 0

    iou_matrix = compute_iou_matrix(a_dets, a_trks)
    scores = np.repeat(a_dets[:, -1][:, np.newaxis], a_trks.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], dx_curr.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * a_vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > a_iou_thre).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = solve_linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections: List[int] = []
    for d, _ in enumerate(a_dets):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers: List[int] = []
    for t, _ in enumerate(a_trks):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches: List[np.ndarray] = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < a_iou_thre:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        final_matches = np.empty((0, 2), dtype=int)
    else:
        final_matches = np.concatenate(matches, axis=0)

    return final_matches, np.array(unmatched_detections), np.array(unmatched_trackers)
