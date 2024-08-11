"""FusionBox

This module implements a post-processing FusionBox to apply filtering and aggregation on the inputs from several
person detectors to finalize the detections.
"""

# region Imported Dependencies
import copy
from typing import Union

import numpy as np

from brain.util.cv.shape import Size
from brain.util.cv.shape.bx import (
    CoordStatus,
    ConfidenceStatus,
    SizeStatus,
    BBox2DNestedList,
    BBox2DList,
)
from brain.util.ml.util import BaseModel


# endregion Imported Dependencies


class FusionBox(BaseModel):
    """FusionBox

    This class implements a post-processing FusionBox to apply filtering and aggregation on the inputs
    from several person detectors.

    Attributes:
        overlap_thre (float): Threshold for determining if two bounding boxes overlap.
        conf_thre (float): Confidence threshold for filtering low-confidence detections.
        top_k_thre (int): Threshold for selecting top-k detections.
        min_size_thre (Size): Minimum size threshold for bounding boxes.
        margin_thre (Size): Margin threshold for adjusting bounding boxes.
    """

    def __init__(
        self,
        a_overlap_thre: float,
        a_conf_thre: float,
        a_top_k_thre: int,
        a_min_size_thre: Size,
        a_margin_thre: Size,
        a_name: str = "FusionBox",
    ):
        """Initialize the FusionBox post-processing module.

        Args:
            a_overlap_thre (float): Threshold for determining if two bounding boxes overlap.
            a_conf_thre (float): Confidence threshold for filtering low-confidence detections.
            a_top_k_thre (int): Threshold for selecting top-k detections.
            a_min_size_thre (Size): Minimum size threshold for bounding boxes.
            a_margin_thre (Size): Margin threshold for adjusting bounding boxes.
            a_name (str, optional): Name of the FusionBox instance. Defaults to "FusionBox".
        """
        super().__init__(a_name)
        self.overlap_thre: float = a_overlap_thre
        self.conf_thre: float = a_conf_thre
        self.top_k_thre: int = a_top_k_thre
        self.min_size_thre: Size = a_min_size_thre
        self.margin_thre: Size = a_margin_thre

    def _filter_top_k(self, a_boxes: np.ndarray, a_scores: np.ndarray) -> np.ndarray:
        """Filter top-k boxes based on their scores.

        Args:
            a_boxes (np.ndarray): Array of bounding boxes.
            a_scores (np.ndarray): Array of confidence scores corresponding to each bounding box.

        Returns:
            np.ndarray: Array of top-k bounding boxes.
        """
        sorted_indices = np.argsort(a_scores)[::-1]
        top_k_indices = sorted_indices[: self.top_k_thre]
        top_k_boxes = a_boxes[top_k_indices]
        return top_k_boxes

    def _filter_low_confidence(
        self, a_boxes: np.ndarray, a_scores: np.ndarray
    ) -> np.ndarray:
        """Filter boxes with confidence scores below a threshold.

        Args:
            a_boxes (np.ndarray): Array of bounding boxes.
            a_scores (np.ndarray): Array of confidence scores corresponding to each bounding box.

        Returns:
            np.ndarray: Array of filtered bounding boxes with confidence scores above or equal to the threshold.
        """
        confident_indices = np.where(a_scores >= self.conf_thre)
        filtered_boxes = a_boxes[confident_indices]
        return filtered_boxes

    def _marginate(self, a_box: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """Apply margin to a bounding box.

        Args:
            a_box (Union[list, np.ndarray]): Bounding box represented as a list or NumPy array.

        Returns:
            Union[list, np.ndarray]: Bounding box with margin applied.
        """
        width = a_box[2] - a_box[0]
        height = a_box[3] - a_box[1]

        margin_factor_width = width * self.margin_thre.width
        margin_factor_height = height * self.margin_thre.height

        a_box[0] = a_box[0] - margin_factor_width
        a_box[1] = a_box[1] - margin_factor_height
        a_box[2] = a_box[2] + margin_factor_width
        a_box[3] = a_box[3] + margin_factor_height
        return a_box

    def _is_overlap(
        self,
        a_first_box: Union[list, np.ndarray],
        a_second_box: Union[list, np.ndarray],
    ) -> bool:
        """Check if two bounding boxes overlap.

        Args:
            a_first_box (Union[list, np.ndarray]): First bounding box.
            a_second_box (Union[list, np.ndarray]): Second bounding box.

        Returns:
            bool: True if the bounding boxes overlap with a certain threshold, False otherwise.
        """
        x_left = max(a_first_box[0], a_second_box[0])
        y_top = max(a_first_box[1], a_second_box[1])
        x_right = min(a_first_box[2], a_second_box[2])
        y_bottom = min(a_first_box[3], a_second_box[3])

        if x_right <= x_left or y_bottom <= y_top:
            status = False
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (a_first_box[2] - a_first_box[0]) * (
                a_first_box[3] - a_first_box[1]
            )
            percentage = intersection_area / box1_area

            if percentage >= self.overlap_thre:
                status = True
            else:
                status = False
        return status

    def _aggregate(self, a_boxes: np.ndarray) -> np.ndarray:
        """Aggregate overlapping bounding boxes.

        This method aggregates overlapping bounding boxes by merging them into larger bounding boxes
        based on their overlap and average confidence scores.

        Args:
            a_boxes (np.ndarray): Array of bounding boxes in the format [[x1, y1, x2, y2, score], ...].

        Returns:
            np.ndarray: Aggregated bounding boxes after merging.
        """

        # Sort boxes by y1
        boxes = a_boxes.tolist()
        boxes = sorted(boxes, key=lambda x: x[1])

        tmp_box = None
        while True:
            merge_count = 0
            used_indices = []
            new_boxes = []
            # Loop over boxes
            for i, box1 in enumerate(boxes):
                scores = [box1[-1]]
                # Apply margin on box1
                margined_box1 = self._marginate(a_box=box1)
                for j, box2 in enumerate(boxes):
                    # If box1 has already been used just continue
                    if i in used_indices or j <= i:
                        continue

                    # Apply margin on box2
                    margined_box2 = self._marginate(a_box=box2)

                    # Merge box1 and box2 if they overlap
                    if self._is_overlap(
                        a_first_box=margined_box1, a_second_box=margined_box2
                    ) or self._is_overlap(
                        a_first_box=margined_box2, a_second_box=margined_box1
                    ):
                        scores.append(box2[-1])
                        tmp_box = [
                            min(box1[0], box2[0]),
                            min(box1[1], box2[1]),
                            max(box1[2], box2[2]),
                            max(box1[3], box2[3]),
                        ]
                        used_indices.append(j)
                        merge_count += 1

                    if tmp_box:
                        box1 = tmp_box

                score = sum(scores) / len(scores)
                if tmp_box:
                    new_boxes.append(tmp_box + [score])
                elif i not in used_indices:
                    new_boxes.append(box1)

                used_indices.append(i)
                tmp_box = None

            # Stop loop if there is no merge anymore
            if merge_count == 0:
                break

            boxes = copy.deepcopy(new_boxes)

        if len(new_boxes) == 0:
            new_boxes = np.empty(shape=(0, 5))
        else:
            new_boxes = np.array(new_boxes)

        return new_boxes

    def infer(self, a_persons: BBox2DNestedList, a_image_size: Size) -> BBox2DList:
        """Perform inference on the aggregated detections.

        This method performs inference on the aggregated detections to filter top-k boxes,
        instantiate BBox2DList objects, and remove invalid persons.

        Args:
            a_persons (BBox2DNestedList): BatchPerson2DList containing detections from multiple detectors.
            a_image_size (Size): Size of the input image.

        Returns:
            BBox2DList: A list of detected persons after post-processing.
        """

        # Convert all detections into [[x1, y1, x2, y2, score], ...] formatted boxes array
        boxes = a_persons.to_xyxys()

        # Filter confident boxes
        confident_boxes = self._filter_low_confidence(
            a_boxes=boxes, a_scores=boxes[:, 4]
        )

        # Aggregate boxes
        merged_boxes = self._aggregate(a_boxes=confident_boxes)

        # Filter top-k boxes
        top_k_boxes = self._filter_top_k(
            a_boxes=merged_boxes, a_scores=merged_boxes[:, 4]
        )

        # Instantiate Persons
        persons = BBox2DList.from_xyxys(
            a_coordinates=top_k_boxes,
            a_img_size=a_image_size,
            a_conf_thre=self.conf_thre,
            a_min_size_thre=self.min_size_thre,
            a_do_validate=True,
        )

        # Remove invalid persons
        persons.remove(
            a_status=[
                CoordStatus.ENTIRELY_OOB,
                CoordStatus.INVALID_COORDINATES,
                ConfidenceStatus.NOT_CONFIDENT,
                SizeStatus.INVALID,
            ]
        )

        # Clamp Persons bounding boxes
        persons.clamp()

        # Remove invalid persons
        persons.remove(
            a_status=[
                CoordStatus.INVALID_COORDINATES,
                SizeStatus.INVALID,
            ]
        )
        return persons
