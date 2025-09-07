"""OC-SORT Multi-Segmented-Instance Tracker
"""

# region Imported Dependencies
import dataclasses
import uuid
from itertools import compress
from typing import List, Optional, Tuple

import numpy as np

from aib.cv.img import Image2D
from aib.cv.vid import Frame2D
from aib.legacy.trk.OCSORT import OCSORT
from aib.legacy.trk.util import TrackedSegBBox2DDict, TrackedSegBBox2D
from .tracklet import PopulationDict, State, KFTarget
from aib.legacy.seg import SegBBox2DList, SegBBox2D
from ..OCSORT import associate, linear_assignment


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
@dataclasses.dataclass
class DetMeta:
    mask: Image2D
    label: int


# TODO(doc): Complete the document of following class
class OCSORTSeg(OCSORT):

    def __init__(
        self,
        a_population_ids: List[uuid.UUID],
        a_det_thre: float,
        a_num_st_thre: int,
        a_max_prediction_age: Optional[int] = 30,
        a_min_update_age: Optional[int] = 3,
        a_iou_thre: Optional[float] = 0.3,
        a_delta_time: Optional[int] = 3,
        a_assoc_fun: Optional[str] = "iou",
        a_inertia: Optional[float] = 0.2,
        a_use_byte: Optional[bool] = False,
        a_name: str = "OCSORTSeg",
    ):
        super().__init__(
            a_population_ids,
            a_det_thre,
            a_num_st_thre,
            a_max_prediction_age,
            a_min_update_age,
            a_iou_thre,
            a_delta_time,
            a_assoc_fun,
            a_inertia,
            a_use_byte,
            a_name,
        )

    @property
    def tracked_population(self) -> PopulationDict:
        return self._tracked_population

    def _preproc(self, a_boxes: SegBBox2DList, a_frame: Frame2D) -> tuple:
        (
            dets,
            trks,
            velocities,
            k_observations,
            dets_second,
            inds_second,
            last_boxes,
        ) = super()._preproc(a_boxes=a_boxes, a_frame=a_frame)

        dets_meta = []
        for box in a_boxes:
            dets_meta.append(DetMeta(mask=box.mask, label=box.label))
        dets_meta_second = list(compress(dets_meta, inds_second))

        return (
            dets,
            dets_meta,
            trks,
            velocities,
            k_observations,
            dets_second,
            dets_meta_second,
            last_boxes,
        )

    def _update_matched_targets(
        self,
        a_matched_pairs: np.ndarray,
        a_dets: np.ndarray,
        a_dets_meta: List[DetMeta],
        a_frame: Frame2D,
    ):
        for m in a_matched_pairs:
            # Index of matched detected box and tracked target
            det_idx = m[0]
            target_idx = m[1]

            box = SegBBox2D.from_xyxys(
                a_coordinates=a_dets[det_idx, :],
                a_mask=a_dets_meta[det_idx].mask.data,
                a_label=a_dets_meta[det_idx].label,
                a_do_validate=False,
            )
            # Update matched target
            self.tracked_population[a_frame.video_id][target_idx].update(a_state=State(a_box=box))

    def _byte_associate(
        self,
        a_dets_second: np.ndarray,
        a_dets_meta_second: List[DetMeta],
        a_unmatched_targets_idx: np.ndarray,
        a_targets: np.ndarray,
        a_frame: Frame2D,
    ) -> np.ndarray:
        unmatched_trks = a_unmatched_targets_idx

        # BYTE association
        if self._use_byte and len(a_dets_second) > 0 and a_unmatched_targets_idx.shape[0] > 0:
            u_trks = a_targets[a_unmatched_targets_idx]
            iou_left = self._associate(a_dets_second, u_trks)  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_thre:
                """
                NOTE: by using a lower threshold, e.g., self.a_iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], a_unmatched_targets_idx[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_thre:
                        continue

                    box = SegBBox2D.from_xyxys(
                        a_coordinates=a_dets_second[det_ind, :],
                        a_mask=a_dets_meta_second[det_ind].mask.data,
                        a_label=a_dets_meta_second[det_ind].label,
                        a_do_validate=False,
                    )
                    # Update target
                    self.tracked_population[a_frame.video_id][trk_ind].update(a_state=State(a_box=box))

                    # Add the target index to remove target
                    to_remove_trk_indices.append(trk_ind)

                unmatched_trks = np.setdiff1d(a_unmatched_targets_idx, np.array(to_remove_trk_indices))
        return unmatched_trks

    def _associate_unmatched_dets_targets(
        self,
        a_dets: np.ndarray,
        a_dets_meta: List[DetMeta],
        a_last_boxes: np.ndarray,
        a_unmatched_dets_idx: np.ndarray,
        a_unmatched_targets_idx: np.ndarray,
        a_frame: Frame2D,
    ) -> Tuple[np.ndarray, np.ndarray]:
        unmatched_dets = a_unmatched_dets_idx
        unmatched_trks = a_unmatched_targets_idx

        if a_unmatched_dets_idx.shape[0] > 0 and a_unmatched_targets_idx.shape[0] > 0:
            left_dets = a_dets[a_unmatched_dets_idx]
            left_trks = a_last_boxes[a_unmatched_targets_idx]
            iou_left = self._associate(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_thre:
                """
                NOTE: by using a lower threshold, e.g., self.a_iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = (
                        a_unmatched_dets_idx[m[0]],
                        a_unmatched_targets_idx[m[1]],
                    )
                    if iou_left[m[0], m[1]] < self.iou_thre:
                        continue

                    box = SegBBox2D.from_xyxys(
                        a_coordinates=a_dets[det_ind, :],
                        a_mask=a_dets_meta[det_ind].mask.data,
                        a_label=a_dets_meta[det_ind].label,
                        a_do_validate=False,
                    )
                    # Update target
                    self.tracked_population[a_frame.video_id][trk_ind].update(a_state=State(a_box=box))
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(a_unmatched_dets_idx, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(a_unmatched_targets_idx, np.array(to_remove_trk_indices))

        return unmatched_dets, unmatched_trks

    def _create_target(
        self,
        a_unmatched_dets_idx: np.ndarray,
        a_dets: np.ndarray,
        a_dets_meta: List[DetMeta],
        a_frame: Frame2D,
    ) -> None:
        # Create targets on the unmatched detections
        for i in a_unmatched_dets_idx:
            # Extract Feature Vector
            box = SegBBox2D.from_xyxys(
                a_coordinates=a_dets[i, :],
                a_mask=a_dets_meta[i].mask.data,
                a_label=a_dets_meta[i].label,
                a_do_validate=False,
            )

            # Create target
            target: KFTarget = KFTarget(
                a_state=State(a_box=box),
                a_num_st_thre=self.num_st_thre,
                a_delta_time=self.delta_time,
            )
            self.tracked_population[a_frame.video_id].append(target)

    def _create_output(self, a_frame: Frame2D) -> TrackedSegBBox2DDict:
        # Create a boxes list
        tracked_objects: TrackedSegBBox2DDict = TrackedSegBBox2DDict()

        i = len(self.tracked_population[a_frame.video_id])
        for trk in reversed(self.tracked_population[a_frame.video_id]):

            if (trk.statistics.post_update_prediction_age < 1) and (
                trk.statistics.post_prediction_update_age >= self.min_update_age
            ):
                # Create object
                object = TrackedSegBBox2D(
                    a_id=trk.id,
                    a_timestamp=trk.timestamp,
                    a_p1=trk.state.box.p1,
                    a_p2=trk.state.box.p2,
                    a_score=trk.state.box.score,
                    a_mask=trk.state.box.mask,
                    a_label=trk.state.box.label,
                    a_img_size=a_frame.size,
                    a_strict=trk.state.box.strict,
                    a_conf_thre=trk.state.box.conf_thre,
                    a_min_size_thre=trk.state.box.min_size_thre,
                    a_do_validate=False,
                    a_name=trk.name,
                )
                # Clamp bounding box
                object.clamp()

                # Add bbox into the list
                tracked_objects.append(a_key=object.id, a_value=object)
            i -= 1

            # Remove dead targets
            if trk.statistics.post_update_prediction_age > self.max_prediction_age:
                self.tracked_population[a_frame.video_id].pop(i)
        return tracked_objects

    def infer(self, a_boxes: SegBBox2DList, a_frame: Frame2D) -> TrackedSegBBox2DDict:
        tracked_boxes: TrackedSegBBox2DDict = TrackedSegBBox2DDict()

        if a_boxes is not None and len(a_boxes) > 0:
            # Preprocess
            (
                dets,
                dets_meta,
                trks,
                velocities,
                k_observations,
                dets_second,
                dets_meta_second,
                last_boxes,
            ) = self._preproc(a_boxes=a_boxes, a_frame=a_frame)

            # 1- First round of association
            matched, unmatched_dets, unmatched_trks = associate(
                dets,
                trks,
                self.iou_thre,
                velocities,
                k_observations,
                self._inertia,
            )

            # Update matched pairs
            self._update_matched_targets(
                a_matched_pairs=matched,
                a_dets=dets,
                a_dets_meta=dets_meta,
                a_frame=a_frame,
            )

            # 2- Second round of associaton by OCR
            unmatched_trks = self._byte_associate(
                a_dets_second=dets_second,
                a_dets_meta_second=dets_meta_second,
                a_unmatched_targets_idx=unmatched_trks,
                a_targets=trks,
                a_frame=a_frame,
            )

            # 3- Third round of association
            unmatched_dets, unmatched_trks = self._associate_unmatched_dets_targets(
                a_dets=dets,
                a_dets_meta=dets_meta,
                a_last_boxes=last_boxes,
                a_unmatched_dets_idx=unmatched_dets,
                a_unmatched_targets_idx=unmatched_trks,
                a_frame=a_frame,
            )

            # Update unmatched targets
            self._update_unmatched_targets(a_unmatched_targets_idx=unmatched_trks, a_frame=a_frame)

            # Create and initialise new target for unmatched detections
            self._create_target(
                a_unmatched_dets_idx=unmatched_dets,
                a_dets=dets,
                a_dets_meta=dets_meta,
                a_frame=a_frame,
            )

            # Create current frame output
            tracked_boxes = self._create_output(a_frame=a_frame)
        return tracked_boxes
