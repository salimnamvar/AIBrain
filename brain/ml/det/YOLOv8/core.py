""" YOLOv8 Person Detector

This module implements a Person Detector based on the YOLOv8 architecture using OpenVino as its backend.
"""

# region Imported Dependencies
from typing import List

import numpy as np
import torch
from ultralytics.utils import ops

from brain.cv.img import Image2D
from brain.cv.proc import letterbox
from brain.cv.shape import Size
from brain.cv.shape.bx import CoordStatus, ConfidenceStatus, SizeStatus, BBox2DList
from brain.ml.det.util import OVObjDetModel


# endregion Imported Dependencies


class YOLOv8(OVObjDetModel):
    """YOLOv8 Person Detector

    This class implements a Person Detector based on the YOLOv8 architecture using OpenVino as its backend.

    Attributes:
        For inherited attribute details, see OVObjectDetectorModel class documentation.
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
    ):
        """Constructor of YOLOv8 Person Detector

        Initializes a new instance of the YOLOv8 Person Detector class.

        Args:
            a_name (str): The name of the YOLOv8 model.
            a_mdl_path (str): The path to the YOLOv8 model.
            a_mdl_device (str): The device on which the YOLOv8 model runs.
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
        """Pre-process input image for YOLOv8 model.

        Args:
            a_image (Image2D): The input image.

        Returns:
            np.ndarray: The pre-processed image array.

        Raises:
            TypeError: If the YOLOv8 model is not loaded successfully.
        """
        self.validate_mdl()

        # Resize Image
        resized_image: Image2D = letterbox(a_image=a_image, a_new_shape=self.mdl_inp_size)[0]

        # Convert HWC to CHW array
        image = resized_image.data.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        image = image.astype(np.float32)  # uint8 to fp32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        # add batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def _postproc(self, a_preds: np.ndarray, a_image: Image2D) -> BBox2DList:
        """Post-process predictions from YOLOv8 model.

        Args:
            a_preds (np.ndarray): The raw predictions from the YOLOv8 model.
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: A list of 2D persons.

        Raises:
            TypeError: If the YOLOv8 model is not loaded successfully.
        """
        self.validate_mdl()

        persons: BBox2DList = BBox2DList()
        if a_preds is not None and len(a_preds):
            # Apply NMS on Predictions and have [[cx, cy, h, w, conf, label], ...]
            preds = ops.non_max_suppression(
                prediction=torch.from_numpy(a_preds),
                conf_thres=self.conf_thre,
                iou_thres=self.nms_thre,
                classes=[0],
                agnostic=False,
                max_det=self.top_k_thre,
                nc=80,
            )[0]

            # Scale Boxes Coordinates and have [[x, y, x, y, conf, label], ...]
            preds[:, :4] = ops.scale_boxes(
                img1_shape=(self.mdl_inp_size.height, self.mdl_inp_size.width),
                boxes=preds[:, :4],
                img0_shape=(a_image.size.height, a_image.size.width),
            ).round()

            # Instantiate Persons
            persons = BBox2DList.from_xyxys(
                a_coordinates=preds[:, :-1],
                a_img_size=a_image.size,
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

    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        """Run inference on the YOLOv8 model.

        Args:
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: A list of 2D persons detected in the image.

        Raises:
            TypeError: If the YOLOv8 model is not loaded successfully.
        """
        self.validate_mdl()

        # Pre-process input image
        proc_input = self._preproc(a_image=a_image)

        # Model inference
        preds = self.mdl.infer_new_request(proc_input)[0]

        # Post-process predictions
        persons = self._postproc(a_preds=preds, a_image=a_image)

        return persons
