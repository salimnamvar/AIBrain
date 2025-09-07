"""Machine Learning - Object Tracking - OCSORT Core

This module implements the OCSORT tracking algorithm for multi-object and multi-view tracking.
It provides the OCSORT class, which manages the lifecycle of object tracks, including detection
association, track prediction, update, and management of track states. The tracker supports
various association metrics (IoU, GIoU, CIoU, DIoU, distance), BYTE association, and velocity-based
association weighting. The class is designed to be generic over bounding box and target types,
and integrates with a broader computer vision and tracking utility framework.

Classes:
    OCSORT:
        Implements the OCSORT tracking algorithm with configurable association metrics,
        observation window, warmup, TTL, and BYTE association support.

Type Variables:
    IOT: TODO
    BoxT: Type variable for bounding box types.
    TargetT: Type variable for target types.
"""

from typing import Any, Callable, Generic, List, Literal, Optional, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt

from aib.cnt.io import QueueIO
from aib.cv.geom.box import AnyBox, AnyBoxList, FloatBox
from aib.cv.img.frame import Frame2D
from aib.cv.img.image import Image2D
from aib.ml.trk.ocsort.utils.assoc import (
    associate,
    compute_ciou_matrix,
    compute_diou_matrix,
    compute_dist_matrix,
    compute_giou_matrix,
    compute_iou_matrix,
    solve_linear_assignment,
)
from aib.ml.trk.ocsort.utils.ent import EntityDict
from aib.ml.trk.ocsort.utils.tgt import AnyTarget, FloatTarget, Target, TargetDict, TargetNestedDict
from aib.ml.trk.utils.b_trk_mdl import BaseTrkModel

IOT = TypeVar(
    "IOT",
    bound=QueueIO[None | Frame2D, None | Tuple[Frame2D, EntityDict]],
    default=QueueIO[None | Frame2D, None | Tuple[Frame2D, EntityDict]],
)
BoxT = TypeVar("BoxT", bound=AnyBox, default=FloatBox)
TargetT = TypeVar("TargetT", bound=AnyTarget, default=FloatTarget)


class OCSORT(BaseTrkModel[IOT], Generic[IOT, BoxT, TargetT]):
    """OCSORT Tracker

    This class implements the OCSORT tracking algorithm for multi-object and multi-view tracking.

    Attributes:
        _obs_size (int): Size of the observation window.
        _obs_lookback_step (int): Lookback step for observations.
        _ttl (int): Time-to-live for tracks.
        _warmup (int): Warmup period for the tracker.
        _iou_thre (float): IoU threshold for association.
        _assoc (Literal['iou', 'giou', 'ciou', 'diou', 'dist']): Association metric.
        _associate_func (Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]]):
            The association function used for tracking.
        _vdc_weight (float): Weight for velocity-based association.
        _use_byte (bool): Whether to use BYTE association.
        _byte_weight (float): Weight for BYTE association.
        _targets (TargetNestedDict[TargetT]): Dictionary of targets indexed by source IDs.
    """

    def __init__(
        self,
        a_obs_size: int = 7,
        a_obs_lookback_step: int = 3,
        a_ttl: int = 30,
        a_warmup: int = 3,
        a_iou_thre: float = 0.3,
        a_assoc: Literal['iou', 'giou', 'ciou', 'diou', 'dist'] = "iou",
        a_vdc_weight: float = 0.2,
        a_use_byte: bool = False,
        a_src_ids: Optional[Tuple[int, ...]] = None,
        a_conf_thre: Optional[float] = None,
        a_id: Optional[int] = None,
        a_name: str = 'OCSORT',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OCSORT.

        Args:
            a_conf_thre (float): Confidence threshold for detections.
            a_obs_size (int): Size of the observation window.
            a_obs_lookback_step (int): Lookback step for observations.
            a_ttl (int): Time-to-live for tracks.
            a_warmup (int): Warmup period for the tracker.
            a_iou_thre (float): IoU threshold for association.
            a_assoc (Literal['iou', 'giou', 'ciou', 'diou', 'dist']): Association metric.
            a_vdc_weight (float): Weight for velocity-based association.
            a_use_byte (bool): Whether to use BYTE association.
            a_src_ids (Optional[Tuple[int, ...]]): Source IDs for the tracks.
            a_model_file (Optional[str | bytes | object | Path]): Path to the model file.
            a_id (Optional[int]): Unique identifier for the tracker.
            a_name (str): Name of the tracker.
            a_enable_profile (bool): Whether to enable profiling.
            a_enable_cfg (bool): Whether to enable configuration.
            a_enable_log (bool): Whether to enable logging.
        """
        super().__init__(
            a_src_ids=a_src_ids,
            a_conf_thre=a_conf_thre,
            a_model_uri=None,
            a_model_version=None,
            a_model_size=None,
            a_model_config=None,
            a_model_in_layers=None,
            a_model_out_layers=None,
            a_backend_core=None,
            a_data_size=None,
            a_infer_timeout=None,
            a_infer_trial=1,
            a_device='AUTO',
            a_precision='FP32',
            a_call_mode='sync',
            a_io_mode='args',
            a_proc_mode='online',
            a_backend='sys',
            a_conc_mode=None,
            a_max_workers=None,
            a_io=None,
            a_stop_event=None,
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._obs_size: int = a_obs_size
        self._obs_lookback_step: int = a_obs_lookback_step
        self._ttl: int = a_ttl
        self._warmup: int = a_warmup
        self._iou_thre: float = a_iou_thre
        self._assoc: Literal['iou', 'giou', 'ciou', 'diou', 'dist'] = a_assoc
        assoc_func_map = {
            "iou": compute_iou_matrix,
            "giou": compute_giou_matrix,
            "ciou": compute_ciou_matrix,
            "diou": compute_diou_matrix,
            "dist": compute_dist_matrix,
        }
        self._associate_func: Callable[
            [npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]
        ] = assoc_func_map[self._assoc]
        self._vdc_weight: float = a_vdc_weight
        self._use_byte: bool = a_use_byte
        self._targets: TargetNestedDict[TargetT] = TargetNestedDict[TargetT]()
        src_ids = self._src_ids or (0,)
        for key_ in src_ids:
            self._targets[key_] = TargetDict[TargetT]()

    def preproc(
        self,
        a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
        a_boxes: AnyBoxList,
        a_src_id: Optional[int] = None,
        a_step_timestamp: Optional[float] = None,
        a_step_id: Optional[int] = None,
    ) -> Tuple[
        int,
        float,
        int,
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float32],
        AnyBoxList,
    ]:
        """Preprocess the input frame and boxes.

        This function extracts relevant information from the input frame and boxes,
        preparing them for further processing.

        Args:
            a_image: The input frame (2D image or array).
            a_boxes: The bounding boxes to process.
            a_src_id: The source ID of the frame.
            a_step_timestamp: The timestamp of the step.
            a_step_id: The ID of the step.

        Returns:
            Tuple containing:
                - src_id (int): Source identifier for the frame
                - step_timestamp (float): Timestamp of the current step
                - step_id (int): Unique identifier for the step
                - dets (npt.NDArray[np.float32]): High-confidence detections [x1, y1, x2, y2, score]
                - trks (npt.NDArray[np.float32]): Predicted target positions [x1, y1, x2, y2, 0]
                - vels (npt.NDArray[np.float32]): Velocity vectors for active targets
                - k_obs (npt.NDArray[np.float32]): Historical observations for targets
                - dets_second (npt.NDArray[np.float32]): Low-confidence detections [x1, y1, x2, y2, score]
                - inds_second (npt.NDArray[np.int32]): Indices of low-confidence detections
                - last_boxes (npt.NDArray[np.float32]): Last known bounding boxes for targets
                - boxes_second (AnyBoxList): Low-confidence detection boxes in original format

        Raises:
            ValueError: If any of the required frame information is missing.
        """

        def _get_image_info(
            a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
            a_src_id: Optional[int] = None,
            a_step_timestamp: Optional[float] = None,
            a_step_id: Optional[int] = None,
        ) -> Tuple[int, float, int]:
            if isinstance(a_image, (Frame2D)):
                src_id = a_image.src_id
                step_timestamp = a_image.timestamp
                step_id = a_image.id
            else:
                src_id = a_src_id
                step_timestamp = a_step_timestamp
                step_id = a_step_id

            # Check for None values and raise errors
            if src_id is None:
                raise ValueError("src_id cannot be None")
            if step_timestamp is None:
                raise ValueError("step_timestamp cannot be None")
            if step_id is None:
                raise ValueError("step_id cannot be None")
            return src_id, step_timestamp, step_id

        def _parse_detections(
            a_dets: npt.NDArray[np.floating | np.integer],
        ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            if a_dets.shape[1] == 5:
                # Format: [x1, y1, x2, y2, confidence_score]
                scores = a_dets[:, 4]
                bboxes = a_dets[:, :4]
            elif a_dets.shape[1] == 6:
                if np.all(a_dets[:, 5] == a_dets[:, 5].astype(int)):
                    # Format: [x1, y1, x2, y2, confidence_score, class_label]
                    scores = a_dets[:, 4]
                    bboxes = a_dets[:, :4]
                else:
                    # Format: [x1, y1, x2, y2, objectness_score, class_confidence]
                    scores = a_dets[:, 4] * a_dets[:, 5]
                    bboxes = a_dets[:, :4]
            else:
                # Default: assume YOLO format with objectness * class_confidence
                scores = a_dets[:, 4] * a_dets[:, 5]
                bboxes = a_dets[:, :4]
            return bboxes.astype(np.float32), scores.astype(np.float32)

        def _filter_detections(
            a_boxes: AnyBoxList, a_boxes_arr: npt.NDArray[np.float32], a_scores: npt.NDArray[np.float32]
        ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], AnyBoxList, npt.NDArray[np.int32]]:
            dets = np.concatenate((a_boxes_arr, np.expand_dims(a_scores, axis=-1)), axis=1)
            if self._conf_thre is not None:
                inds_low = scores > 0.1
                inds_high = scores < self._conf_thre
                inds_second = np.logical_and(inds_low, inds_high)
                remain_inds = scores > self._conf_thre
                dets_second = dets[inds_second]
                boxes_second = a_boxes[inds_second]
                dets = dets[remain_inds]
            else:
                dets_second = np.empty((0, dets.shape[1]), dtype=np.float32)
                boxes_second = type(a_boxes)()
                inds_second = np.array([], dtype=np.int32)
            return dets, dets_second, boxes_second, inds_second

        def _get_target_predictions(a_src_id: int) -> npt.NDArray[np.float32]:
            trks = np.zeros((len(self._targets[a_src_id]), 5), dtype=np.float32)
            to_del: list[int] = []
            # Predict next positions for all existing targets
            for t, trk in enumerate(trks):
                pred_box = self._targets[a_src_id].get_by_index(t).predict()
                if pred_box is None:
                    trk[:] = [np.nan, np.nan, np.nan, np.nan, 0]
                    to_del.append(t)
                else:
                    pos = pred_box.to_xyxy()
                    trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            # Clean up invalid predictions and remove failed targets
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                self._targets[a_src_id].pop_by_index(t)
            return trks

        def _get_target_velocities(a_src_id: int) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            # Get velocity vectors for all active targets
            vels = np.array([trk.stats.velocity for trk in self._targets[a_src_id].values()], dtype=np.float32)

            # Get last known bounding boxes for targets
            last_boxes = np.array(
                [trk.obs[-1].to_xyxy()[:5] for trk in self._targets[a_src_id].values()], dtype=np.float32
            )
            return vels, last_boxes

        def _get_observations(a_src_id: int) -> npt.NDArray[np.float32]:
            tmp: list[npt.NDArray[np.floating | np.integer]] = []
            for trk in self._targets[a_src_id].values():
                tmp.append(
                    trk.obs.get_prev_obs(a_curr_age=trk.stats.predict_count, a_lookback_steps=self._obs_lookback_step)
                )
            prev_obs = np.array(tmp, dtype=np.float32)
            return prev_obs

        src_id, step_timestamp, step_id = _get_image_info(
            a_image=a_image, a_src_id=a_src_id, a_step_timestamp=a_step_timestamp, a_step_id=a_step_id
        )
        dets_xyxy = a_boxes.to_xyxy()
        boxes_xyxy, scores = _parse_detections(dets_xyxy)
        dets, dets_second, boxes_second, inds_second = _filter_detections(
            a_boxes=a_boxes, a_boxes_arr=boxes_xyxy, a_scores=scores
        )
        trks = _get_target_predictions(a_src_id=src_id)
        vels, last_boxes = _get_target_velocities(a_src_id=src_id)
        prev_obs = _get_observations(a_src_id=src_id)
        return (
            src_id,
            step_timestamp,
            step_id,
            dets,
            trks,
            vels,
            prev_obs,
            dets_second,
            inds_second,
            last_boxes,
            boxes_second,
        )

    def _associate(
        self,
        a_dets: npt.NDArray[np.float32],
        a_trks: npt.NDArray[np.float32],
        a_vels: npt.NDArray[np.float32],
        a_prev_obs: npt.NDArray[np.float32],
        a_dets_second: npt.NDArray[np.float32],
        a_boxes_second: AnyBoxList,
        a_boxes: AnyBoxList,
        a_last_boxes: npt.NDArray[np.float32],
        a_src_id: int,
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        """Associate detections with tracks.

        Args:
            a_dets (npt.NDArray[np.float32]): Detections to associate.
            a_trks (npt.NDArray[np.float32]): Tracks to associate with.
            a_vels (npt.NDArray[np.float32]): Velocities of the tracks.
            a_prev_obs (npt.NDArray[np.float32]): Previous observations of the tracks.
            a_dets_second (npt.NDArray[np.float32]): Second set of detections.
            a_boxes_second (AnyBoxList): Second set of bounding boxes.
            a_boxes (AnyBoxList): First set of bounding boxes.
            a_last_boxes (npt.NDArray[np.float32]): Last known bounding boxes of the tracks.
            a_src_id (int): Source ID of the frame.

        Returns:
            Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
                - Matched pairs of detections and tracks.
                - Unmatched detections.
                - Unmatched tracks.
        """

        def _byte_associate(
            a_dets_second: npt.NDArray[np.float32],
            a_boxes_second: AnyBoxList,
            a_unmatched_trks: npt.NDArray[np.integer],
            a_trks: npt.NDArray[np.float32],
            a_src_id: int,
        ) -> npt.NDArray[np.integer]:
            unmatched_trks = a_unmatched_trks
            if self._use_byte and len(a_dets_second) > 0 and a_unmatched_trks.shape[0] > 0:
                u_trks = a_trks[a_unmatched_trks]
                iou_left = self._associate_func(a_dets_second, u_trks)
                iou_left = np.array(iou_left)
                if iou_left.max() > self._iou_thre:
                    matched_indices = solve_linear_assignment(-iou_left)
                    to_remove_trk_indices: list[int] = []
                    for m in matched_indices:
                        det_ind, trk_ind = m[0], a_unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self._iou_thre:
                            continue

                        self._targets[a_src_id].get_by_index(trk_ind).update(
                            a_box=a_boxes_second[int(det_ind)], a_obs_lookback_step=self._obs_lookback_step
                        )

                        to_remove_trk_indices.append(int(trk_ind))

                    unmatched_trks = np.setdiff1d(a_unmatched_trks, np.array(to_remove_trk_indices))
            return unmatched_trks

        def _fallback_associate(
            a_dets: npt.NDArray[np.float32],
            a_boxes: AnyBoxList,
            a_last_boxes: npt.NDArray[np.float32],
            a_unmatched_dets: npt.NDArray[np.integer],
            a_unmatched_trks: npt.NDArray[np.integer],
            a_src_id: int,
        ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
            unmatched_dets: npt.NDArray[np.integer] = a_unmatched_dets
            unmatched_trks: npt.NDArray[np.integer] = a_unmatched_trks
            if a_unmatched_dets.shape[0] > 0 and a_unmatched_trks.shape[0] > 0:
                left_dets = a_dets[a_unmatched_dets]
                left_trks = a_last_boxes[a_unmatched_trks]
                iou_left = self._associate_func(left_dets, left_trks)
                iou_left = np.array(iou_left)
                if iou_left.max() > self._iou_thre:
                    rematched_indices = solve_linear_assignment(-iou_left)
                    to_remove_det_indices: List[int] = []
                    to_remove_trk_indices: List[int] = []
                    for m in rematched_indices:
                        det_ind, trk_ind = (a_unmatched_dets[m[0]], a_unmatched_trks[m[1]])
                        if iou_left[m[0], m[1]] < self._iou_thre:
                            continue

                        self._targets[a_src_id].get_by_index(trk_ind).update(
                            a_box=a_boxes[int(det_ind)], a_obs_lookback_step=self._obs_lookback_step
                        )
                        to_remove_det_indices.append(int(det_ind))
                        to_remove_trk_indices.append(int(trk_ind))
                    unmatched_dets = np.setdiff1d(a_unmatched_dets, np.array(to_remove_det_indices))
                    unmatched_trks = np.setdiff1d(a_unmatched_trks, np.array(to_remove_trk_indices))
            return unmatched_dets, unmatched_trks

        matched_pairs, unmatched_dets, unmatched_trks = associate(
            a_dets=a_dets,
            a_trks=a_trks,
            a_iou_thre=self._iou_thre,
            a_vels=a_vels,
            a_prev_obs=a_prev_obs,
            a_vdc_weight=self._vdc_weight,
        )
        unmatched_trks = _byte_associate(
            a_dets_second=a_dets_second,
            a_boxes_second=a_boxes_second,
            a_unmatched_trks=unmatched_trks,
            a_trks=a_trks,
            a_src_id=a_src_id,
        )
        unmatched_dets, unmatched_trks = _fallback_associate(
            a_dets=a_dets,
            a_boxes=a_boxes,
            a_last_boxes=a_last_boxes,
            a_unmatched_dets=unmatched_dets,
            a_unmatched_trks=unmatched_trks,
            a_src_id=a_src_id,
        )
        return matched_pairs, unmatched_dets, unmatched_trks

    def _assign(
        self,
        a_matched_pairs: npt.NDArray[np.integer],
        a_unmatched_dets: npt.NDArray[np.integer],
        a_unmatched_trks: npt.NDArray[np.integer],
        a_boxes: AnyBoxList,
        a_src_id: int,
        a_step_timestamp: float,
        a_step_id: int,
    ) -> EntityDict[BoxT]:
        """Assign detections to tracks

        Args:
            a_matched_pairs (npt.NDArray[np.integer]): Matched pairs of detections and tracks.
            a_unmatched_dets (npt.NDArray[np.integer]): Unmatched detections.
            a_unmatched_trks (npt.NDArray[np.integer]): Unmatched tracks.
            a_boxes (AnyBoxList): List of bounding boxes.
            a_src_id (int): Source ID of the frame.
            a_step_timestamp (float): Timestamp of the frame.
            a_step_id (int): ID of the frame.

        Returns:
            EntityDict[BoxT]: Dictionary of tracked entities.
        """

        def _update(
            a_matched_pairs: npt.NDArray[np.integer],
            a_unmatched_trks: npt.NDArray[np.integer],
            a_boxes: AnyBoxList,
            a_src_id: int,
        ) -> None:
            # Update matched targets
            for m_pair in a_matched_pairs:
                det_idx = int(m_pair[0])
                trk_idx = int(m_pair[1])
                self._targets[a_src_id].get_by_index(trk_idx).update(
                    a_box=a_boxes[det_idx], a_obs_lookback_step=self._obs_lookback_step
                )

            # Update unmatched targets
            for unm_id in a_unmatched_trks:
                self._targets[a_src_id].get_by_index(int(unm_id)).update(
                    a_box=None, a_obs_lookback_step=self._obs_lookback_step
                )

        def _insert(
            a_unmatched_dets: npt.NDArray[np.integer],
            a_boxes: AnyBoxList,
            a_src_id: int,
            a_step_timestamp: float,
            a_step_id: int,
        ) -> None:
            for ind in a_unmatched_dets:
                trk_id = self._targets[a_src_id].get_new_id()
                target: Target[BoxT] = Target[BoxT].create(
                    a_id=trk_id,
                    a_box=a_boxes[int(ind)],
                    a_obs_size=self._obs_size,
                    a_step_id=a_step_id,
                    a_step_timestamp=a_step_timestamp,
                )
                self._targets[a_src_id][trk_id] = cast(TargetT, target)

        def _select(a_src_id: int) -> EntityDict[BoxT]:
            entities = EntityDict[BoxT]()
            for trk in self._targets[a_src_id].values():
                if (trk.stats.age_since_update < 1) and (trk.stats.age_since_predict >= self._warmup):
                    entities[trk.id] = cast(BoxT, trk.obs[-1])
            return entities

        def _delete(a_src_id: int) -> None:
            to_del: List[int] = []
            for trk_id, trk in self._targets[a_src_id].items():
                if trk.stats.age_since_update > self._ttl:
                    to_del.append(trk_id)

            for trk_id in to_del:
                self._targets[a_src_id].pop_by_key(trk_id)

        # Update matched and unmatched targets
        _update(a_matched_pairs=a_matched_pairs, a_unmatched_trks=a_unmatched_trks, a_boxes=a_boxes, a_src_id=a_src_id)
        # Insert new targets
        _insert(
            a_unmatched_dets=a_unmatched_dets,
            a_boxes=a_boxes,
            a_src_id=a_src_id,
            a_step_id=a_step_id,
            a_step_timestamp=a_step_timestamp,
        )
        # Select active targets
        entities = _select(a_src_id=a_src_id)
        # Delete old targets
        _delete(a_src_id=a_src_id)
        return entities

    def infer(
        self,
        a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
        a_boxes: Optional[AnyBoxList] = None,
        a_src_id: Optional[int] = None,
        a_step_timestamp: Optional[float] = None,
        a_step_id: Optional[int] = None,
    ) -> EntityDict[BoxT]:
        """OCSORT Tracker Inference

        Args:
            a_image (Image2D | Frame2D | npt.NDArray[np.uint8]): Input image for inference.
            a_boxes (Optional[AnyBoxList]): Detected boxes for the current frame.
            a_src_id (Optional[int]): Source ID for the current image.
            a_step_timestamp (Optional[float]): Timestamp for the current step(frame).
            a_step_id (Optional[int]): Step ID for the current step(frame).

        Returns:
            EntityDict[BoxT]: Dictionary of tracked entities.
        """
        entities: EntityDict[BoxT] = EntityDict[BoxT]()
        if a_boxes is not None and len(a_boxes) > 0:
            (
                src_id,
                frame_timestamp,
                frame_id,
                dets,
                trks,
                vels,
                prev_obs,
                dets_second,
                inds_second,
                last_boxes,
                boxes_second,
            ) = self.preproc(
                a_image=a_image,
                a_boxes=a_boxes,
                a_src_id=a_src_id,
                a_step_timestamp=a_step_timestamp,
                a_step_id=a_step_id,
            )

            matched_pairs, unmatched_dets, unmatched_trks = self._associate(
                a_dets=dets,
                a_trks=trks,
                a_vels=vels,
                a_prev_obs=prev_obs,
                a_dets_second=dets_second,
                a_boxes_second=boxes_second,
                a_src_id=src_id,
                a_boxes=a_boxes,
                a_last_boxes=last_boxes,
            )

            entities = self._assign(
                a_matched_pairs=matched_pairs,
                a_unmatched_dets=unmatched_dets,
                a_unmatched_trks=unmatched_trks,
                a_boxes=a_boxes,
                a_src_id=src_id,
                a_step_timestamp=frame_timestamp,
                a_step_id=frame_id,
            )
        return entities
