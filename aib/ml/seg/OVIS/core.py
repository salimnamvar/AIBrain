"""OVIS Instance Segmentor

This module implements an instance segmentation model based on the OpenVino Intel-based architecture.
"""

# region Imported Dependencies
from typing import List

import cv2
import numpy as np
from openvino.runtime.utils.data_helpers import OVDict

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.cv.shape.bx import CoordStatus, ConfidenceStatus, SizeStatus
from aib.ml.seg.util import OVInstSegModel, SegBBox2DList
from aib.ml.seg.util import SegBBox2D


# endregion Imported Dependencies


class OVIS(OVInstSegModel):
    """OVIS Instance Segmentor

    This class implements an instance segmentation model based on the OpenVino architecture.

    Attributes:
        Inherits attributes from the parent class `OVInstSegModel`.
    """

    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
        a_conf_thre: float,
        a_nms_thre: float,
        a_top_k_thre: int,
        a_min_size_thre: Size,
        a_classes: List[int] = None,
    ) -> None:
        """
        Initializes a new instance of the model.

        Args:
            a_name (str): The name of the model.
            a_mdl_path (str): The path to the model.
            a_mdl_device (str): The device on which the model runs.
            a_conf_thre (float): Confidence threshold for predictions.
            a_nms_thre (float): Non-maximum suppression threshold.
            a_top_k_thre (int): Top-k threshold for predictions.
            a_min_size_thre (Size): Minimum size threshold for bounding boxes.
            a_classes (List[int], optional):
                A list of class indices to consider. If None, all classes will be considered.
        """
        super().__init__(
            a_name=a_name,
            a_mdl_path=a_mdl_path,
            a_mdl_device=a_mdl_device,
            a_conf_thre=a_conf_thre,
            a_nms_thre=a_nms_thre,
            a_top_k_thre=a_top_k_thre,
            a_min_size_thre=a_min_size_thre,
            a_classes=a_classes,
        )

    def _preproc(self, a_image: Image2D) -> np.ndarray:
        """
        Pre-processes input data before inference.

        Args:
            a_image (Image2D): The input image.

        Returns:
            np.ndarray: Pre-processed input data.
        """
        self.validate_mdl()

        # Resize Image
        image = cv2.resize(
            a_image.data,
            self.mdl_inp_size.to_tuple(),
            interpolation=cv2.INTER_AREA,
        )

        # Convert HWC to CHW array
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, 0)

        return image

    def _postproc(self, a_preds: OVDict, a_image: Image2D) -> SegBBox2DList:
        """
        Post-processes model predictions.

        Args:
            a_preds (OVDict): Model predictions.
            a_image (Image2D): The input image.

        Returns:
            SegBBox2DList: A list of segmented bounding boxes.
        """
        self.validate_mdl()

        instances: SegBBox2DList = SegBBox2DList()
        if a_preds is not None:
            # Data extraction
            labels = a_preds["labels"]
            boxes = a_preds["boxes"]
            masks = a_preds["masks"]

            # Filter selected classes
            if self.classes is not None:
                sel_indices = np.where(labels == self.classes)[0]
                labels = labels[sel_indices]
                boxes = boxes[sel_indices, :]
                masks = masks[sel_indices, :]

            # Apply NMS
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=boxes[:, 0:4],
                scores=boxes[:, 4],
                score_threshold=self.conf_thre,
                nms_threshold=self.nms_thre,
                top_k=self.top_k_thre,
            )
            nms_boxes = boxes[nms_indices, :]
            nms_masks = masks[nms_indices, :]
            nms_labels = labels[nms_indices]

            # Scale boxes
            scaled_boxes = nms_boxes.copy()
            scaled_boxes[:, [0, 2]] *= a_image.width / self.mdl_inp_size.width
            scaled_boxes[:, [1, 3]] *= a_image.height / self.mdl_inp_size.height

            for box, mask, label in zip(scaled_boxes, nms_masks, nms_labels):
                # Scale Masks
                scaled_mask = cv2.resize(mask, (int(box[2]) - int(box[0]), int(box[3]) - int(box[1])))

                instances.append(
                    a_item=SegBBox2D.from_xyxys(
                        a_coordinates=box,
                        a_mask=scaled_mask,
                        a_label=label,
                        a_img_size=a_image.size,
                        a_conf_thre=self.conf_thre,
                        a_min_size_thre=self.min_size_thre,
                        a_do_validate=True,
                    )
                )

            # Remove invalid persons
            instances.remove(
                a_status=[
                    CoordStatus.ENTIRELY_OOB,
                    CoordStatus.INVALID_COORDINATES,
                    ConfidenceStatus.NOT_CONFIDENT,
                    SizeStatus.INVALID,
                ]
            )

            # Clamp Persons bounding boxes
            instances.clamp()

            # Remove invalid persons
            instances.remove(
                a_status=[
                    CoordStatus.INVALID_COORDINATES,
                    SizeStatus.INVALID,
                ]
            )
        return instances

    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        """
        Runs inference on the model.

        Args:
            a_image (Image2D): The input image.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            SegBBox2DList: A list of segmented bounding boxes.

        Raises:
            RuntimeError: If the model is not loaded or initialized.
        """
        self.validate_mdl()

        # Pre-process input image
        proc_input = self._preproc(a_image=a_image)

        # Model inference
        preds = self.mdl.infer_new_request(proc_input)

        # Post-process predictions
        persons = self._postproc(a_preds=preds, a_image=a_image)

        return persons
