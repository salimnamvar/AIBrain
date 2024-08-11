"""OC-SORT Association Utilities

This module provides utility functions for association in the OC-SORT tracker.

Functions:
    - `iou_batch(bboxes1, bboxes2)`: Computes the Intersection over Union (IoU) between two sets of bounding boxes.
    - `giou_batch(bboxes1, bboxes2)`: Computes the Generalized IoU (GIoU) between two sets of bounding boxes.
    - `diou_batch(bboxes1, bboxes2)`: Computes the Distance IoU (DIoU) between two sets of bounding boxes.
    - `ciou_batch(bboxes1, bboxes2)`: Computes the Complete IoU (CIoU) between two sets of bounding boxes.
    - `ct_dist(bboxes1, bboxes2)`: Measures the center distance between two sets of bounding boxes.
    - `speed_direction_batch(dets, tracks)`: Computes the normalized speed and direction between detections and tracks.
    - `linear_assignment(cost_matrix)`: Assigns the optimal pairings using linear assignment.
    - `associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)`:
        Associates detections to tracked objects based on IoU.
    - `associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight)`:
        Associates detections to trackers considering velocity direction consistency.
    - `associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight)`:
        Associates detections to trackers with category-aware considerations for KITTI dataset.

Note:
    - For detailed information on IoU variants (GIoU, DIoU, CIoU), refer to the
      paper: https://arxiv.org/pdf/1902.09630.pdf.
    - The `vdc_weight` parameter in `associate` and `associate_kitti` functions controls the influence of velocity
      direction consistency in the association process.

"""


# region Imported Dependencies
import numpy as np

# endregion Imported Dependencies


def iou_batch(bboxes1, bboxes2):
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        bboxes1 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        bboxes2 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: IoU values between each pair of bounding boxes.

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        This function uses the formula for IoU: Area of Intersection / Area of Union.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def giou_batch(bboxes1, bboxes2):
    """
    Compute the Generalized Intersection over Union (GIoU) between two sets of bounding boxes.

    Args:
        bboxes1 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        bboxes2 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: GiOU values between each pair of bounding boxes.

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on GiOU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    Compute the Distance-IoU (DIoU) between two sets of bounding boxes.

    Args:
        bboxes1 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        bboxes2 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: DIoU values between each pair of bounding boxes.

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on DIoU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    iou = wh / union
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2):
    """
    Compute the Complete-IoU (CIoU) between two sets of bounding boxes.

    Args:
        bboxes1 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        bboxes2 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: CIoU values between each pair of bounding boxes.

    Note:
        The input bounding boxes are assumed to be in the format [x1, y1, x2, y2].
        For details on CIoU, refer to: https://arxiv.org/pdf/1902.09630.pdf
    """
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    iou = wh / union

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.0
    h1 = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
    Measure the center distance between two sets of bounding boxes.

    Args:
        bboxes1 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        bboxes2 (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: Center distance values between each pair of bounding boxes.

    Note:
        This is a coarse implementation, and it may not be suitable for precise association.
        The returned values are normalized to the range (0, 1).
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist  # resize to (0,1)


def speed_direction_batch(dets, tracks):
    """
    Calculate the speed direction between detection bounding boxes and tracks.

    Args:
        dets (numpy.ndarray): Detection bounding boxes in the format [x1, y1, x2, y2].
        tracks (numpy.ndarray): Track bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        Tuple of numpy.ndarray: Tuple containing two arrays (dy, dx) representing the normalized
        speed direction vectors. The sizes are (num_tracks, num_detections).

    Note:
        The speed direction vectors are calculated based on the center points of the bounding boxes.
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def linear_assignment(cost_matrix):
    """
    Perform linear assignment using either the lap library or scipy's linear_sum_assignment.

    Args:
        cost_matrix (numpy.ndarray): Cost matrix for the assignment problem.

    Returns:
        numpy.ndarray: Array containing pairs of indices representing the assignment.

    Note:
        The lap library is preferred if available, as it can handle rectangular cost matrices.
    """
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate(
    detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight
):
    """
    Associate detections with existing trackers based on IOU and velocity direction.

    Args:
        detections (numpy.ndarray): Array of bounding boxes representing detections.
        trackers (numpy.ndarray): Array of bounding boxes representing existing trackers.
        iou_threshold (float): IOU threshold for considering a match.
        velocities (numpy.ndarray): Array of velocities for each tracker.
        previous_obs (numpy.ndarray): Previous observations used to calculate speed and direction.
        vdc_weight (float): Weight for velocity direction cost.

    Returns:
        Tuple:
            - matched_indices (numpy.ndarray): Array containing matched indices between detections and trackers.
            - unmatched_detections (numpy.ndarray): Array containing indices of unmatched detections.
            - unmatched_trackers (numpy.ndarray): Array containing indices of unmatched trackers.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
