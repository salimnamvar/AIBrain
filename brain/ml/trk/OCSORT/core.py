"""OC-SORT Multi-Object Tracker

The `OCSORT` module serves as Multi-Camera Multi-Object Tracker, building upon the foundation of the
Observation-Centric SORT (OC-SORT) algorithm. This tracker is specifically designed for real-time object tracking
and across multiple cameras in complex surveillance scenarios.

Note:
    For detailed information on the algorithm, please refer to
        - Paper: https://arxiv.org/abs/2203.14360
        - Code: https://github.com/noahcao/OC_SORT
"""

# region Imported Dependencies
import uuid
from typing import List, Optional, Tuple
import numpy as np
from brain.cv.shape.bx import BBox2DList
from brain.cv.vid import Frame2D
from brain.ml.reid.util import BaseReidModel
from brain.ml.trk import TrackedBBox2DDict, TrackedBBox2D, BBoxTrkModel
from brain.ml.trk.OCSORT import (
    PopulationDict,
    KFTargetList,
    iou_batch,
    giou_batch,
    ciou_batch,
    diou_batch,
    ct_dist,
    k_previous_obs,
    associate,
    State,
    linear_assignment,
    KFTarget,
)

# endregion Imported Dependencies


class OCSORT(BBoxTrkModel):
    """Multi-Camera OC-SORT Multi-Object Tracker

    Attributes:
        det_thre (float): Detection threshold for object detection.
        max_prediction_age (int): Maximum age for predictions before a tracklet is considered invalid and removed.
        min_update_age (int): Minimum age required for a tracklet before updating its predictions.
        iou_thre (float): IOU threshold for object association.
        delta_time (int): Time window for considering previous observations during association.
        assoc_fun (str): Association function used for matching, e.g., "iou," "giou," "ciou," "diou," or "ct_dist."
        inertia (float): Inertia factor for smoothing the velocity of predictions.
        use_byte (bool): Boolean flag indicating whether to use BYTE for object association.
        population_ids (List[uuid.UUID]): List of unique identifiers for each camera or population.
        num_st_thre (int): Number of states considered for matching.
        tracked_population (Dict[uuid.UUID, List[KFTarget]]):
            Dictionary to store populations of tracked targets for each camera or population.
    """

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
        a_reid: Optional[BaseReidModel] = None,
        a_name: str = "OCSORT",
    ):
        """
        Initializes a new Tracker instance.

        Args:
            a_population_ids (List[uuid.UUID]): List of UUIDs representing population IDs.
            a_det_thre (float): Detection threshold.
            a_num_st_thre (int): Number of states threshold.
            a_max_prediction_age (Optional[int]): Maximum age for predictions. Default is 30.
            a_min_update_age (Optional[int]): Minimum age for updates. Default is 3.
            a_iou_thre (Optional[float]): IOU threshold. Default is 0.3.
            a_delta_time (Optional[int]): Time difference threshold. Default is 3.
            a_assoc_fun (Optional[str]): Association function. Default is "iou".
            a_inertia (Optional[float]): Inertia parameter. Default is 0.2.
            a_use_byte (Optional[bool]): Flag indicating whether to use byte. Default is False.
        """
        super().__init__(a_name=a_name, a_reid=a_reid)
        self.det_thre: float = a_det_thre
        self.max_prediction_age: int = a_max_prediction_age
        self.min_update_age: int = a_min_update_age
        self.iou_thre: float = a_iou_thre
        self.delta_time: int = a_delta_time
        self._associate: callable = None
        self.assoc_fun: str = a_assoc_fun
        self.inertia: float = a_inertia
        self.use_byte: bool = a_use_byte
        self.population_ids: List[uuid.UUID] = a_population_ids
        self.num_st_thre: int = a_num_st_thre

        # Create Population
        self._create_populations()

    # region Attributes
    @property
    def num_st_thre(self) -> int:
        """
        Getter for the number of states considered for matching.

        Returns:
            int: Number of states considered for matching.
        """
        return self._num_st_thre

    @num_st_thre.setter
    def num_st_thre(self, a_num_st_thre: int) -> None:
        """
        Setter for the number of states considered for matching.

        Args:
            a_num_st_thre (int): Number of states to set.

        Raises:
            TypeError: If `a_num_st_thre` is not an integer.
        """
        if a_num_st_thre is None or not isinstance(a_num_st_thre, int):
            raise TypeError("The `a_num_st_thre` must be a `int`.")
        self._num_st_thre: int = a_num_st_thre

    @property
    def det_thre(self) -> float:
        """
        Getter method for the detection threshold.

        Returns:
            float: The current detection threshold.
        """
        return self._det_thre

    @det_thre.setter
    def det_thre(self, a_det_thre: float) -> None:
        """
        Setter method for the detection threshold.

        Args:
            a_det_thre (float): The new detection threshold.

        Raises:
            TypeError: If `a_det_thre` is None or not a float.
        """
        if a_det_thre is None or not isinstance(a_det_thre, float):
            raise TypeError("The `a_det_thre` must be a `float`.")
        self._det_thre: float = a_det_thre

    @property
    def max_prediction_age(self) -> int:
        """
        Getter for the maximum prediction age.

        Returns:
            int: The maximum prediction age.
        """
        return self._max_prediction_age

    @max_prediction_age.setter
    def max_prediction_age(self, a_max_prediction_age: int) -> None:
        """
        Setter for the maximum prediction age.

        Args:
            a_max_prediction_age (int): The maximum prediction age to set.

        Raises:
            TypeError: If `a_max_prediction_age` is not an integer.
        """
        if a_max_prediction_age is None or not isinstance(a_max_prediction_age, int):
            raise TypeError("The `a_max_prediction_age` must be a `int`.")
        self._max_prediction_age: int = a_max_prediction_age

    @property
    def min_update_age(self) -> int:
        """
        Getter method for the minimum update age.

        Returns:
            int: The current minimum update age.
        """
        return self._min_update_age

    @min_update_age.setter
    def min_update_age(self, a_min_update_age: int) -> None:
        """
        Setter method for the minimum update age.

        Args:
            a_min_update_age (int): The new minimum update age.

        Raises:
            TypeError: If `a_min_update_age` is None or not an int.
        """
        if a_min_update_age is None or not isinstance(a_min_update_age, int):
            raise TypeError("The `min_update_age` must be a `int`.")
        self._min_update_age: int = a_min_update_age

    @property
    def iou_thre(self) -> float:
        """
        Getter for the Intersection over Union (IoU) threshold.

        Returns:
            float: The IoU threshold.
        """
        return self._iou_thre

    @iou_thre.setter
    def iou_thre(self, a_iou_thre: float) -> None:
        """
        Setter for the Intersection over Union (IoU) threshold.

        Args:
            a_iou_thre (float): The IoU threshold to set.

        Raises:
            TypeError: If `a_iou_thre` is not a float.
        """
        if a_iou_thre is None or not isinstance(a_iou_thre, float):
            raise TypeError("The `a_iou_thre` must be a `float`.")
        self._iou_thre: float = a_iou_thre

    @property
    def delta_time(self) -> int:
        """
        Getter for the delta time parameter.

        Returns:
            int: The delta time value.
        """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, a_delta_time: int) -> None:
        """
        Setter for the delta time parameter.

        Args:
            a_delta_time (int): The delta time value to set.

        Raises:
            TypeError: If `a_delta_time` is not an int.
        """
        if a_delta_time is None or not isinstance(a_delta_time, int):
            raise TypeError("The `a_delta_time` must be a `int`.")
        self._delta_time: int = a_delta_time

    @property
    def assoc_fun(self) -> str:
        """
        Getter for the association function name.

        Returns:
            str: The association function name.
        """
        return self._assoc_fun

    @assoc_fun.setter
    def assoc_fun(self, a_assoc_fun: int) -> None:
        """
        Setter for the association function name.

        Args:
            a_assoc_fun (str): The association function name to set.

        Raises:
            TypeError: If `a_assoc_fun` is not a string.
            ValueError: If the provided `a_assoc_fun` is not a valid function name.
        """
        functions_mapping = {
            "iou": iou_batch,
            "giou": giou_batch,
            "ciou": ciou_batch,
            "diou": diou_batch,
            "ct_dist": ct_dist,
        }
        if a_assoc_fun is None or not isinstance(a_assoc_fun, str):
            raise TypeError("The `a_assoc_fun` must be a `str`.")

        if a_assoc_fun not in functions_mapping:
            raise ValueError(
                f"Invalid value for `a_assoc_fun`. "
                f"Expected one of {list(functions_mapping.keys())}, but got {a_assoc_fun}."
            )
        self._assoc_fun: str = a_assoc_fun
        self._associate: callable = functions_mapping[a_assoc_fun]

    @property
    def inertia(self) -> float:
        """
        Getter for the inertia value.

        Returns:
            float: The inertia value.
        """
        return self._inertia

    @inertia.setter
    def inertia(self, a_inertia: float) -> None:
        """
        Setter for the inertia value.

        Args:
            a_inertia (float): The inertia value to set.

        Raises:
            TypeError: If `a_inertia` is not a float.
        """
        if a_inertia is None or not isinstance(a_inertia, float):
            raise TypeError("The `a_inertia` must be a `float`.")
        self._inertia: float = a_inertia

    @property
    def use_byte(self) -> float:
        """
        Getter for the use_byte property.

        Returns:
            bool: The use_byte property value.
        """
        return self._use_byte

    @use_byte.setter
    def use_byte(self, a_use_byte: bool) -> None:
        """
        Setter for the use_byte property.

        Args:
            a_use_byte (bool): The value to set for the use_byte property.

        Raises:
            TypeError: If `a_use_byte` is not a bool.
        """
        if a_use_byte is None or not isinstance(a_use_byte, bool):
            raise TypeError("The `a_use_byte` must be a `bool`.")
        self._use_byte: bool = a_use_byte

    @property
    def population_ids(self) -> List[uuid.UUID]:
        """
        Getter for the population_ids property.

        Returns:
            List[uuid.UUID]: The population_ids property value.
        """
        return self._population_ids

    @population_ids.setter
    def population_ids(self, a_population_ids: List[uuid.UUID]) -> None:
        """
        Setter for the population_ids property.

        Args:
            a_population_ids (List[uuid.UUID]): The value to set for the population_ids property.

        Raises:
            TypeError: If `a_population_ids` is not a List of uuid.UUID.
        """
        if (
            a_population_ids is None
            or not isinstance(a_population_ids, list)
            or not all([isinstance(pop, uuid.UUID) for pop in a_population_ids])
        ):
            raise TypeError("The `a_population_ids` must be a `List[uuid.UUID]`.")
        self._population_ids: List[uuid.UUID] = a_population_ids

    @property
    def tracked_population(self) -> PopulationDict:
        """
        Getter method for the tracked population.

        Returns:
            PopulationDict: The current tracked population.
        """
        return self._tracked_population

    @tracked_population.setter
    def tracked_population(self, a_tracked_population: PopulationDict) -> None:
        """
        Setter method for the tracked population.

        Args:
            a_tracked_population (PopulationDict): The new tracked population.

        Raises:
            TypeError: If `a_tracked_population` is None or not a PopulationDict.
        """
        if a_tracked_population is None or not isinstance(a_tracked_population, PopulationDict):
            raise TypeError("The `a_tracked_population` must be a `PopulationDict`.")
        self._tracked_population: PopulationDict = a_tracked_population

    # endregion Attributes

    def _create_populations(self) -> None:
        """Create Population

        Initializes the populations of the tracker, creating a population for each camera.

        Returns:
            None
        """
        self._tracked_population: PopulationDict = PopulationDict()
        for key in self.population_ids:
            self.tracked_population.append(a_key=key, a_value=KFTargetList())

    def _preproc(self, a_boxes: BBox2DList, a_frame: Frame2D) -> tuple:
        """
        Preprocesses the detected boxes and frame data for further tracking steps.

        Args:
            a_boxes (BBox2DList): List of detected boxes.
            a_frame (Frame2D): The current frame.

        Returns:
            tuple: A tuple containing preprocessed data including detections, metadata, tracks, velocities,
                   k observations, second round detections, second round metadata, and last boxes.
        """
        a_det_bboxes = a_boxes.to_xyxys()

        # post_process detections
        if a_det_bboxes.shape[1] == 5:
            scores = a_det_bboxes[:, 4]
            bboxes = a_det_bboxes[:, :4]
        else:
            scores = a_det_bboxes[:, 4] * a_det_bboxes[:, 5]
            bboxes = a_det_bboxes[:, :4]  # x1y1x2y2

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self._det_thre
        inds_second = np.logical_and(inds_low, inds_high)  # self._det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        boxes_second = a_boxes[inds_second]

        remain_inds = scores > self._det_thre
        dets = dets[remain_inds]

        # get predicted locations from existing populations.
        trks = np.zeros((len(self.tracked_population[a_frame.video_id]), 5))
        to_del = []
        for t, trk in enumerate(trks):
            prediction = self.tracked_population[a_frame.video_id][t].predict()
            if prediction.box is None:
                trk[:] = [np.nan, np.nan, np.nan, np.nan, 0]
                to_del.append(t)
            else:
                pos = prediction.box.to_xyxy()
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracked_population[a_frame.video_id].pop(t)

        velocities = np.array(
            [
                (trk.statistics.velocity if trk.statistics.velocity is not None else np.array((0, 0)))
                for trk in self.tracked_population[a_frame.video_id]
            ]
        )

        last_boxes = np.array([trk.state.box.to_xyxys() for trk in self.tracked_population[a_frame.video_id]])

        tmp = []
        for trk in self.tracked_population[a_frame.video_id]:
            tmp.append(k_previous_obs(trk.states, trk.statistics.prediction_age, self.delta_time))

        k_observations = np.array(tmp)

        return dets, trks, velocities, k_observations, dets_second, inds_second, last_boxes, boxes_second

    def _update_matched_targets(
        self,
        a_matched_pairs: np.ndarray,
        a_dets: BBox2DList,
        a_frame: Frame2D,
    ) -> None:
        """Update Matched Targets

        Updates the tracked targets that are matched in the process of association.

        Args:
            a_matched_pairs (np.ndarray): Array containing pairs of matched detected boxes and tracked targets.
            a_dets (np.ndarray): Array of detections.
            a_frame (Frame2D): The current frame.
        """
        for m in a_matched_pairs:
            # Index of matched detected box and tracked target
            det_idx = m[0]
            target_idx = m[1]

            # UPDATE tracking entity
            self.tracked_population[a_frame.video_id][target_idx].update(a_state=State(a_box=a_dets[det_idx]))

    def _byte_associate(
        self,
        a_dets_second: np.ndarray,
        a_boxes_second: BBox2DList,
        a_unmatched_targets_idx: np.ndarray,
        a_targets: np.ndarray,
        a_frame: Frame2D,
    ) -> np.ndarray:
        """Byte Association

        Second round of association focusing on unmatched tracked targets to match them with remaining detected objects
        with lower confidence.

        Args:
            a_dets_second (np.ndarray): Array of low confidence detections.
            a_unmatched_targets_idx (np.ndarray): Array of indices of unmatched tracked targets.
            a_targets (np.ndarray): Array of tracked targets.
            a_frame (Frame2D): The current frame.

        Returns:
            np.ndarray: Array of indices of unmatched tracked targets after association.
        """
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

                    # Update target
                    self.tracked_population[a_frame.video_id][trk_ind].update(
                        a_state=State(a_box=a_boxes_second[det_ind])
                    )

                    # Add the target index to remove target
                    to_remove_trk_indices.append(trk_ind)

                unmatched_trks = np.setdiff1d(a_unmatched_targets_idx, np.array(to_remove_trk_indices))
        return unmatched_trks

    def _associate_unmatched_dets_targets(
        self,
        a_dets: np.ndarray,
        a_boxes: BBox2DList,
        a_last_boxes: np.ndarray,
        a_unmatched_dets_idx: np.ndarray,
        a_unmatched_targets_idx: np.ndarray,
        a_frame: Frame2D,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Associate Unmatched Detected Objects and Targets

        Third round of association focusing on unmatched detected objects with unmatched tracked targets.

        Args:
            a_dets (np.ndarray): Array of detected objects.
            a_last_boxes (np.ndarray): Array of last known bounding boxes of tracked targets.
            a_unmatched_dets_idx (np.ndarray): Array of indices of unmatched detected objects.
            a_unmatched_targets_idx (np.ndarray): Array of indices of unmatched tracked targets.
            a_frame (Frame2D): The current frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of indices of unmatched detected objects and unmatched tracked targets
            after association.
        """
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

                    # Update target
                    self.tracked_population[a_frame.video_id][trk_ind].update(a_state=State(a_box=a_boxes[det_ind]))
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(a_unmatched_dets_idx, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(a_unmatched_targets_idx, np.array(to_remove_trk_indices))

        return unmatched_dets, unmatched_trks

    def _update_unmatched_targets(self, a_unmatched_targets_idx: np.ndarray, a_frame: Frame2D) -> None:
        """Update Unmatched Targets

        Updates the tracked targets that are unmatched in the process of association.

        Args:
            a_unmatched_targets_idx (np.ndarray): Array of indices of unmatched tracked targets.
            a_frame (Frame2D): The current frame.

        Returns:
            None: It does not return any values.
        """
        # Update unmatched targets
        for m in a_unmatched_targets_idx:
            self.tracked_population[a_frame.video_id][m].update(a_state=None)

    def _create_target(
        self,
        a_dets: BBox2DList,
        a_unmatched_dets_idx: np.ndarray,
        a_frame: Frame2D,
    ) -> None:
        """Create Target

        Create new tracking targets based on unmatched detected objects.

        Args:
            a_dets (np.ndarray): Array of detected objects.
            a_unmatched_dets_idx (np.ndarray): Array of indices of unmatched detected objects.
            a_frame (Frame2D): Information about the current frame.
        """

        # Create targets on the unmatched detections
        for i in a_unmatched_dets_idx:
            # Create target
            target: KFTarget = KFTarget(
                a_time=a_frame.time,
                a_state=State(a_box=a_dets[i]),
                a_num_st_thre=self.num_st_thre,
                a_delta_time=self.delta_time,
            )
            self.tracked_population[a_frame.video_id].append(target)

    def _create_output(self, a_frame: Frame2D) -> TrackedBBox2DDict:
        """Create Current Frame's Tracked Targets

        Generates a list of tracked targets as the output of the tracker for the current frame. Additionally, it checks
        whether a target is considered dead and removes it from the population if necessary.

        Args:
            a_frame (Frame2D): The current frame.

        Returns:
            TrackedBBox2DDict: A dictionary of tracked objects.
        """

        # Create a boxes list
        tracked_objects: TrackedBBox2DDict = TrackedBBox2DDict()

        i = len(self.tracked_population[a_frame.video_id])
        for trk in reversed(self.tracked_population[a_frame.video_id]):

            if (trk.statistics.post_update_prediction_age < 1) and (
                trk.statistics.post_prediction_update_age >= self.min_update_age
            ):
                # Create object
                object = TrackedBBox2D(
                    a_id=trk.id,
                    a_name=trk.name,
                    a_timestamp=trk.time,
                    a_p1=trk.state.box.p1,
                    a_p2=trk.state.box.p2,
                    a_score=trk.state.box.score,
                    a_img_size=a_frame.size,
                    a_strict=trk.state.box.strict,
                    a_conf_thre=trk.state.box.conf_thre,
                    a_min_size_thre=trk.state.box.min_size_thre,
                    a_do_validate=False,
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

    def infer(self, a_boxes: BBox2DList, a_frame: Frame2D) -> TrackedBBox2DDict:
        """Inference

        Performs inference on the given detected objects list and current frame.

        Args:
            a_boxes (BBox2DList): List of detected bounding boxes.
            a_frame (Frame2D):The current frame.

        Returns:
            TrackedBBox2DDict: Dictionary containing tracked 2D bounding boxes.
        """

        tracked_boxes: TrackedBBox2DDict = TrackedBBox2DDict()

        if a_boxes is not None and len(a_boxes) > 0:
            # Preprocess
            dets, trks, velocities, k_observations, dets_second, inds_second, last_boxes, boxes_second = self._preproc(
                a_boxes=a_boxes, a_frame=a_frame
            )

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
                a_dets=a_boxes,
                a_frame=a_frame,
            )

            # 2- Second round of association by OCR
            unmatched_trks = self._byte_associate(
                a_dets_second=dets_second,
                a_boxes_second=boxes_second,
                a_unmatched_targets_idx=unmatched_trks,
                a_targets=trks,
                a_frame=a_frame,
            )

            # 3- Third round of association
            unmatched_dets, unmatched_trks = self._associate_unmatched_dets_targets(
                a_dets=dets,
                a_boxes=a_boxes,
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
                a_dets=a_boxes,
                a_frame=a_frame,
            )

            # Create current frame output
            tracked_boxes = self._create_output(a_frame=a_frame)
        return tracked_boxes
