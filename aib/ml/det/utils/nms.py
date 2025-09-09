"""Machine Learning - Object Detection - Non-Maximum Suppression (NMS) Utilities.

This module provides advanced Non-Maximum Suppression (NMS) implementations for
object detection pipelines. It includes:

1. UltralyticsNMS:
    - Supports both axis-aligned and rotated bounding boxes (OBB).
    - Implements probabilistic IoU for rotated boxes.
    - Supports multi-label, class-agnostic, and apriori-label NMS.
    - Highly configurable with confidence thresholds, IoU thresholds, top-k filtering,
      and optional time limits.

2. TorchNMS:
    - Provides standard NMS, Fast-NMS, and batched/class-aware NMS for axis-aligned boxes.
    - Fast-NMS leverages upper-triangular matrix operations for high-speed inference.
    - Batched NMS offsets boxes per class to avoid cross-class suppression.
    - Supports optional custom IoU functions for flexibility.

Classes:
    UltralyticsNMS: NMS utility class for YOLO-style predictions, including rotated and probabilistic boxes.
    TorchNMS: Optimized PyTorch-based NMS utility, including fast and batched NMS methods.
"""

import time
import warnings
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import openvino as ov
import torch
import torchvision


class UltralyticsNMS:
    """Non-Maximum Suppression utility class inspired by Ultralytics YOLO.

    This class provides a collection of static methods for computing
    non-maximum suppression (NMS) on both axis-aligned and rotated
    bounding boxes, with optional support for probabilistic IoU.

    Methods:
        batch_probiou(a_boxes1, a_boxes2, a_eps=1e-7) -> torch.Tensor
            Compute the probabilistic IoU between two sets of rotated bounding boxes.
        get_covariance_matrix(a_boxes) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Compute covariance matrix components for rotated bounding boxes.
        xywh2xyxy(a_boxes) -> torch.Tensor
            Convert bounding boxes from [x_center, y_center, width, height] format
            to [x1, y1, x2, y2] corner format.
        empty_like(a_tensor) -> torch.Tensor | np.ndarray
            Create an empty tensor or array with the same shape and dtype float32 as the input.
        nms(a_preds, a_conf_thre=0.25, a_iou_thre=0.45, a_top_k_thre=300, ...)
            Perform non-maximum suppression on a batch of predictions, supporting
            rotated boxes, multi-label detection, class-agnostic mode, and optional apriori labels.
    """

    @staticmethod
    def batch_probiou(
        a_boxes1: torch.Tensor | npt.NDArray[Any], a_boxes2: torch.Tensor | npt.NDArray[Any], a_eps: float = 1e-7
    ) -> torch.Tensor:
        """Calculate the probabilistic Intersection over Union (IoU) between rotated bounding boxes.

        This method computes the probabilistic IoU between two sets of oriented bounding boxes
        using a Gaussian-based approximation, which accounts for both box rotation and size.
        The resulting score ranges from 0 (no overlap) to 1 (perfect alignment).

        Args:
            a_boxes1 (torch.Tensor | np.ndarray): Bounding boxes of shape (N, 5+) in
                [x_center, y_center, w, h, angle] format. Can be a PyTorch tensor or NumPy array.
            a_boxes2 (torch.Tensor | np.ndarray): Bounding boxes of shape (M, 5+) in
                [x_center, y_center, w, h, angle] format. Can be a PyTorch tensor or NumPy array.
            a_eps (float, optional): Small value to avoid division by zero in calculations. Defaults to 1e-7.

        Returns:
            torch.Tensor: A tensor of shape (N, M) with probabilistic IoU scores between each pair
            of boxes in `a_boxes1` and `a_boxes2`.

        References:
            https://arxiv.org/pdf/2106.06072v1
        """

        boxes1 = torch.from_numpy(a_boxes1) if isinstance(a_boxes1, np.ndarray) else a_boxes1
        boxes2 = torch.from_numpy(a_boxes2) if isinstance(a_boxes2, np.ndarray) else a_boxes2

        x1, y1 = boxes1[..., :2].split(1, dim=-1)
        x1, y1 = cast(torch.Tensor, x1), cast(torch.Tensor, y1)
        x2, y2 = (x.squeeze(-1)[None] for x in boxes2[..., :2].split(1, dim=-1))
        x2, y2 = cast(torch.Tensor, x2), cast(torch.Tensor, y2)
        a1, b1, c1 = UltralyticsNMS.get_covariance_matrix(boxes1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in UltralyticsNMS.get_covariance_matrix(boxes2))

        t1 = (
            ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + a_eps)
        ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + a_eps)) * 0.5
        t3 = (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + a_eps)
            + a_eps
        ).log() * 0.5
        bd = (t1 + t2 + t3).clamp(a_eps, 100.0)
        hd = (1.0 - (-bd).exp() + a_eps).sqrt()
        hd = cast(torch.Tensor, hd)
        return cast(torch.Tensor, 1 - hd)

    @staticmethod
    def get_covariance_matrix(a_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the covariance matrix components for rotated bounding boxes.

        This method generates the covariance matrix components for a set of oriented bounding boxes
        represented in [x_center, y_center, width, height, angle] format. The covariance is derived
        from the width, height, and rotation angle of each box, which is used for probabilistic IoU
        and other geometric calculations. The center coordinates are ignored in this computation.

        Args:
            a_boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes
                in [x_center, y_center, width, height, angle] format.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors corresponding to the
            covariance matrix components:
                - a: variance along the x-axis
                - b: variance along the y-axis
                - c: covariance term representing correlation between x and y axes
        """
        gbbs = torch.cat((a_boxes[:, 2:4].pow(2) / 12, a_boxes[:, 4:]), dim=-1)
        a, b, c = gbbs.split(1, dim=-1)
        a, b, c = cast(torch.Tensor, a), cast(torch.Tensor, b), cast(torch.Tensor, c)
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    @staticmethod
    def xywh2xyxy(a_boxes: torch.Tensor) -> torch.Tensor:
        """Convert bounding boxes from center format to corner format.

        Transforms bounding box coordinates from [x_center, y_center, width, height] format
        to [x1, y1, x2, y2] format, where (x1, y1) is the top-left corner and (x2, y2) is
        the bottom-right corner. This operation is optimized to work per 2-channel slices
        for better performance.

        Args:
            a_boxes (torch.Tensor): A tensor of shape (..., 4) representing bounding boxes
                in [x_center, y_center, width, height] format.

        Returns:
            torch.Tensor: A tensor of the same shape as `a_boxes` with coordinates in
            [x1, y1, x2, y2] format.

        Raises:
            AssertionError: If the last dimension of `a_boxes` is not 4.
        """
        assert a_boxes.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {a_boxes.shape}"
        y = UltralyticsNMS.empty_like(a_boxes)
        xy = a_boxes[..., :2]
        wh = a_boxes[..., 2:] / 2
        y[..., :2] = xy - wh
        y[..., 2:] = xy + wh
        return cast(torch.Tensor, y)

    @staticmethod
    def empty_like(a_tensor: torch.Tensor | npt.NDArray[Any]) -> torch.Tensor | npt.NDArray[Any]:
        """Create an uninitialized tensor or array with the same shape and float32 dtype as the input.

        This function returns an empty (uninitialized) tensor if the input is a `torch.Tensor`
        or an empty array if the input is a `np.ndarray`. The returned object has the same
        shape as the input and a dtype of `float32`.

        Args:
            a_tensor (torch.Tensor | np.ndarray):
                Input tensor or array whose shape will be used to create the empty output.

        Returns:
            torch.Tensor | np.ndarray:
                An uninitialized tensor or array with the same shape as `a_tensor` and dtype `float32`.
        """
        return (
            torch.empty_like(a_tensor, dtype=torch.float32)
            if isinstance(a_tensor, torch.Tensor)
            else np.empty_like(a_tensor, dtype=np.float32)
        )

    @staticmethod
    def nms(
        a_preds: npt.NDArray[Any] | ov.utils.data_helpers.OVDict | Sequence[npt.NDArray[Any]],
        a_conf_thre: float = 0.25,
        a_iou_thre: float = 0.45,
        a_top_k_thre: int = 300,
        a_classes: Optional[Tuple[int, ...]] = None,
        a_agnostic: bool = False,
        a_multi_label: bool = False,
        a_labels: Optional[List[List[Union[npt.NDArray[Any], torch.Tensor]]]] = None,
        a_num_classes: Optional[int] = 0,
        a_pre_nms_top_k: int = 30000,
        a_max_wh: int = 7680,
        a_rotated: bool = False,
        a_use_nms: bool = True,
        a_torch_nms_mode: Literal["custom", "original"] = "original",
        a_max_time: float = 0.05,
        a_break_on_timeout: bool = False,
        a_return_idxs: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Perform non-maximum suppression (NMS) on a batch of prediction results.

        Filters overlapping bounding boxes based on confidence and Intersection-over-Union (IoU)
        thresholds. Supports standard and rotated bounding boxes, multi-label detection, class-agnostic
        NMS, and optional apriori labels. Can return filtered detection indices if required.

        Args:
            a_preds (np.ndarray | OVDict | Sequence[np.ndarray]):
                Model predictions for each image,
                with shape (batch_size, num_boxes, num_classes + 4 + extra). Extra may include masks.
            a_conf_thre (float, optional): Confidence threshold for filtering detections. Must be in [0, 1].
            a_iou_thre (float, optional): IoU threshold for NMS filtering. Must be in [0, 1].
            a_top_k_thre (int, optional): Maximum number of boxes to keep after NMS per image.
            a_classes (Tuple[int, ...], optional): Classes to filter. If None, all classes are considered.
            a_agnostic (bool, optional): Whether to perform class-agnostic NMS.
            a_multi_label (bool, optional): Whether each box can have multiple labels.
            a_labels (List[List[np.ndarray | torch.Tensor]], optional): Apriori labels per image to include in NMS.
            a_num_classes (int, optional): Number of classes. If 0, inferred from predictions.
            a_pre_nms_top_k (int, optional): Maximum number of boxes to consider before NMS.
            a_max_wh (int, optional): Maximum width/height for bounding boxes when performing class-aware NMS.
            a_rotated (bool, optional): Whether bounding boxes are rotated (OBB).
            a_use_nms (bool, optional): Whether to perform NMS. If False, boxes are filtered only by confidence.
            a_torch_nms_mode (Literal['custom', 'original'], optional): Use original torchvision NMS or custom NMS.
            a_max_time (float, optional): Maximum time in seconds per image for NMS before issuing a warning.
            a_break_on_timeout (bool, optional): Whether to stop processing if time limit is exceeded.
            a_return_idxs (bool, optional): Whether to return the indices of kept boxes alongside filtered boxes.

        Returns:
            Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
                - If `a_return_idxs` is False: List of filtered detections per image. Each tensor has shape
                (num_boxes, 6 + extra) with columns: [x1, y1, x2, y2, confidence, class, ...extra].
                - If `a_return_idxs` is True: Tuple of (filtered detections, indices of kept boxes).

        Raises:
            RuntimeError: If the NMS computation fails for any reason.

        Notes:
            - Supports both PyTorch tensors and NumPy arrays as input.
            - For rotated boxes, probabilistic IoU (batch_probiou) is used instead of standard IoU.
            - Extra columns beyond class scores (e.g., masks) are preserved in the output.
        """
        try:
            assert (
                0 <= a_conf_thre <= 1
            ), f"Invalid Confidence threshold {a_conf_thre}, valid values are between 0.0 and 1.0"
            assert 0 <= a_iou_thre <= 1, f"Invalid IoU {a_iou_thre}, valid values are between 0.0 and 1.0"

            if isinstance(a_preds, (list, tuple)):
                preds = torch.tensor(a_preds[0], device='cpu')
            elif isinstance(a_preds, ov.utils.data_helpers.OVDict):
                preds = torch.from_numpy(a_preds[0])
            elif isinstance(a_preds, np.ndarray):
                preds = torch.from_numpy(a_preds)
            else:
                preds = cast(torch.Tensor, a_preds)

            if a_classes is not None:
                classes = torch.tensor(a_classes, device='cpu')
            else:
                classes = a_classes

            if preds.shape[-1] == 6 or not a_use_nms:
                output = [pred[pred[:, 4] > a_conf_thre][:a_top_k_thre] for pred in preds]
                if classes is not None:
                    output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
                return output

            batch_size = preds.shape[0]
            num_classes = a_num_classes or (preds.shape[1] - 4)
            extra = preds.shape[1] - num_classes - 4
            mi = 4 + num_classes
            xc = preds[:, 4:mi].amax(1) > a_conf_thre
            xinds = torch.arange(preds.shape[-1], device=preds.device).expand(batch_size, -1)[..., None]

            multi_label = a_multi_label & (num_classes > 1)

            preds = preds.transpose(-1, -2)
            if not a_rotated:
                preds[..., :4] = UltralyticsNMS.xywh2xyxy(preds[..., :4])

            time_limit = 2.0 + a_max_time * batch_size
            t = time.time()

            output = [torch.zeros((0, 6 + extra), device=preds.device)] * batch_size
            keepi = [torch.zeros((0, 1), device=preds.device)] * batch_size
            for xi, (x, xk) in enumerate(zip(preds, xinds)):
                filt = xc[xi]
                x = x[filt]
                if a_return_idxs:
                    xk = xk[filt]

                if a_labels and len(a_labels[xi]) and not a_rotated:
                    lb = cast(torch.Tensor, a_labels[xi])
                    v = torch.zeros((len(lb), num_classes + extra + 4), device=x.device)
                    v[:, :4] = UltralyticsNMS.xywh2xyxy(lb[:, 1:5])
                    v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
                    x = torch.cat((x, v), 0)

                if not x.shape[0]:
                    continue

                box, cls, mask = x.split((4, num_classes, extra), 1)
                box, cls, mask = cast(torch.Tensor, box), cast(torch.Tensor, cls), cast(torch.Tensor, mask)

                if multi_label:
                    i, j = torch.where(cls > a_conf_thre)
                    x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
                    if a_return_idxs:
                        xk = xk[i]
                else:
                    conf, j = cls.max(1, keepdim=True)
                    filt = conf.view(-1) > a_conf_thre
                    x = torch.cat((box, conf, j.float(), mask), 1)[filt]
                    if a_return_idxs:
                        xk = xk[filt]

                if classes is not None:
                    filt = (x[:, 5:6] == classes).any(1)
                    x = x[filt]
                    if a_return_idxs:
                        xk = xk[filt]

                n = x.shape[0]
                if not n:
                    continue
                if n > a_pre_nms_top_k:
                    filt = x[:, 4].argsort(descending=True)[:a_pre_nms_top_k]
                    x = x[filt]
                    if a_return_idxs:
                        xk = xk[filt]

                c = x[:, 5:6] * (0 if a_agnostic else a_max_wh)
                scores = x[:, 4]
                if a_rotated:
                    boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
                    i = TorchNMS.fast_nms(boxes, scores, a_iou_thre, a_iou_func=UltralyticsNMS.batch_probiou)
                else:
                    boxes = x[:, :4] + c
                    if a_torch_nms_mode == "original":
                        i = torchvision.ops.nms(boxes, scores, a_iou_thre)
                    elif a_torch_nms_mode == "custom":
                        i = TorchNMS.nms(boxes, scores, a_iou_thre)

                i = i[:a_top_k_thre]

                output[xi] = x[i]
                if a_return_idxs:
                    keepi[xi] = xk[i].view(-1)
                if (time.time() - t) > time_limit:
                    warnings.warn(f"UltralyticsNMS.nms time limit {time_limit:.3f}s exceeded")
                    if a_break_on_timeout:
                        break

            return (output, keepi) if a_return_idxs else output
        except Exception as e:
            raise RuntimeError(f"UltralyticsNMS.nms failed. Original error: {e}") from e


class TorchNMS:
    """
    Ultralytics custom Non-Maximum Suppression (NMS) utility class optimized for YOLO.

    This class provides a set of static methods for performing NMS operations on bounding boxes,
    including standard NMS, Fast-NMS, and batched/class-aware NMS. It supports both axis-aligned
    bounding boxes and optional custom IoU functions for flexibility.

    Static Methods:
        box_iou(a_boxes1, a_boxes2, a_eps=1e-7):
            Compute pairwise Intersection-over-Union (IoU) between two sets of boxes.
        fast_nms(a_boxes, a_scores, a_iou_thre, a_use_triu=True, a_iou_func=None):
            Perform Fast Non-Maximum Suppression using upper-triangular matrix optimization.
        nms(a_boxes, a_scores, a_iou_thre):
            Optimized standard NMS with early termination for axis-aligned boxes.
        batched_nms(a_boxes, a_scores, a_idxs, a_iou_thre, a_use_fast_nms=False):
            Perform class-aware batched NMS by offsetting boxes per class index.

    Notes:
        - All NMS methods return indices of boxes to keep.
        - Fast-NMS is useful for large-scale inference and may slightly differ from standard NMS.
        - Batched NMS prevents cross-class suppression by offsetting box coordinates.
    """

    @staticmethod
    def box_iou(a_boxes1: torch.Tensor, a_boxes2: torch.Tensor, a_eps: float = 1e-7) -> torch.Tensor:
        """Compute pairwise Intersection-over-Union (IoU) between two sets of bounding boxes.

        This function calculates the IoU for all pairs of axis-aligned bounding boxes
        provided in `(x1, y1, x2, y2)` format. A small epsilon is added to the denominator
        to avoid division by zero.

        Args:
            a_boxes1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes
                in `[x1, y1, x2, y2]` format.
            a_boxes2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes
                in `[x1, y1, x2, y2]` format.
            a_eps (float, optional): Small value added to the denominator to prevent division
                by zero. Defaults to 1e-7.

        Returns:
            torch.Tensor: A tensor of shape (N, M) containing the pairwise IoU values
            between each box in `a_boxes1` and each box in `a_boxes2`.

        Notes:
            - Input tensors are cast to `float` to ensure accurate IoU computation.
            - IoU is computed as `intersection / (area1 + area2 - intersection)`.

        References:
            https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
        """
        (a1, a2), (b1, b2) = a_boxes1.float().unsqueeze(1).chunk(2, 2), a_boxes2.float().unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + a_eps)

    @staticmethod
    def fast_nms(
        a_boxes: torch.Tensor,
        a_scores: torch.Tensor,
        a_iou_thre: float,
        a_use_triu: bool = True,
        a_iou_func: Optional[Callable[..., torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Perform Fast Non-Maximum Suppression (Fast-NMS) on bounding boxes.

        Fast-NMS is an optimized NMS implementation that leverages upper-triangular
        matrix operations for faster computation. It can optionally use a custom IoU
        function for flexibility (e.g., standard IoU or probabilistic IoU).

        Args:
            a_boxes (torch.Tensor):
                Tensor of shape (N, 4) representing bounding boxes in `[x1, y1, x2, y2]` format.
            a_scores (torch.Tensor): Confidence scores for each box, shape (N,).
            a_iou_thre (float):
                IoU threshold for suppression. Boxes with IoU above this value are suppressed.
            a_use_triu (bool, optional):
                If True, only the upper-triangular part of the IoU matrix is used for suppression. Defaults to True.
            a_iou_func (Callable, optional):
                Function to compute IoU between boxes. If None, defaults to `TorchNMS.box_iou`.

        Returns:
            torch.Tensor: Indices of boxes to keep after Fast-NMS.

        Notes:
            - If no boxes are provided, returns an empty tensor of dtype `int64`.
            - When `a_use_triu` is False, the full IoU matrix is used with scores updated
            to avoid retaining overlapping boxes.
            - This method is suitable for high-speed inference scenarios.

        References:
            - https://arxiv.org/pdf/1904.02689
        """
        if a_iou_func is None:
            a_iou_func = TorchNMS.box_iou
        if a_boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=a_boxes.device)

        sorted_idx = torch.argsort(a_scores, descending=True)
        a_boxes = a_boxes[sorted_idx]
        ious = a_iou_func(a_boxes, a_boxes)
        if a_use_triu:
            ious = ious.triu_(diagonal=1)
            pick = torch.nonzero((ious >= a_iou_thre).sum(0) <= 0).squeeze_(-1)
        else:
            n = a_boxes.shape[0]
            row_idx = torch.arange(n, device=a_boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=a_boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            a_scores[~((ious >= a_iou_thre).sum(0) <= 0)] = 0
            pick = torch.topk(a_scores, a_scores.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(a_boxes: torch.Tensor, a_scores: torch.Tensor, a_iou_thre: float) -> torch.Tensor:
        """Perform optimized Non-Maximum Suppression (NMS) on bounding boxes.

        This method removes overlapping boxes based on confidence scores and an IoU
        threshold, with early termination for efficiency. It produces results equivalent
        to torchvision’s NMS implementation but is fully vectorized with a fast exit
        when no overlaps remain.

        Args:
            a_boxes (torch.Tensor):
                Tensor of shape (N, 4) containing bounding boxes in `[x1, y1, x2, y2]` format.
            a_scores (torch.Tensor): Confidence scores for each box, shape (N,).
            a_iou_thre (float):
                IoU threshold for suppression. Boxes with IoU above this threshold are suppressed.

        Returns:
            torch.Tensor: Indices of boxes to keep after NMS, in the original box order.

        Notes:
            - Early termination is applied when no remaining boxes overlap, which improves
            performance on large box sets.
            - Pre-allocates a tensor for `keep` to avoid dynamic list operations.
            - Suitable for axis-aligned boxes (not rotated).
        """
        if a_boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=a_boxes.device)

        x1, y1, x2, y2 = a_boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        _, order = a_scores.sort(0, descending=True)

        keep = torch.zeros(order.numel(), dtype=torch.int64, device=a_boxes.device)
        keep_idx = 0

        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h

            if inter.sum() == 0:
                remaining_count = rest.numel()
                keep[keep_idx : keep_idx + remaining_count] = rest
                keep_idx += remaining_count
                break

            iou = inter / (areas[i] + areas[rest] - inter)

            mask = iou <= a_iou_thre
            order = rest[mask]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        a_boxes: torch.Tensor,
        a_scores: torch.Tensor,
        a_idxs: torch.Tensor,
        a_iou_thre: float,
        a_use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Perform batched Non-Maximum Suppression (NMS) for class-aware suppression.

        This method applies NMS separately for each class by offsetting boxes according
        to their class indices, preventing boxes from different classes from suppressing
        each other. Supports both standard and Fast-NMS implementations.

        Args:
            a_boxes (torch.Tensor):
                Tensor of shape (N, 4) containing bounding boxes in `[x1, y1, x2, y2]` format.
            a_scores (torch.Tensor): Confidence scores for each box, shape (N,).
            a_idxs (torch.Tensor):
                Class indices for each box, shape (N,). Used to offset boxes for class-aware NMS.
            a_iou_thre (float): IoU threshold for suppression.
            a_use_fast_nms (bool, optional):
                If True, use `fast_nms` implementation; otherwise, use standard `nms`. Defaults to False.

        Returns:
            torch.Tensor: Indices of boxes to keep after batched NMS.

        Notes:
            - Early exit is applied if `a_boxes` is empty.
            - Offsetting by `max_coordinate + 1` ensures boxes from different classes do not
            interfere with each other during NMS.
            - Fast-NMS is more efficient for large numbers of boxes but may slightly differ
            from standard NMS in corner cases.
        """
        if a_boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=a_boxes.device)

        max_coordinate = a_boxes.max()
        offsets = a_idxs.to(a_boxes) * (max_coordinate + 1)
        boxes_for_nms = a_boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, a_scores, a_iou_thre)
            if a_use_fast_nms
            else TorchNMS.nms(boxes_for_nms, a_scores, a_iou_thre)
        )
