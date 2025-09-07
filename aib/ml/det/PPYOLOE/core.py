"""PP-YOLOE+ Person Detector

This module implements a Person Detector based on the PP-YOLOE+ model.
"""

# region Imported Dependencies
from typing import List

import cv2
import numpy as np

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.cv.shape.bx import CoordStatus, ConfidenceStatus, SizeStatus, BBox2DList
from aib.ml.det.util import OVObjDetModel


# endregion Imported Dependencies


class PPYOLOE(OVObjDetModel):
    """PP-YOLOE+ Person Detector

    This class implements a person detector based on the PP-YOLOE+ model.
    It inherits from the OVPrsDetModel class.

    Attributes:
        inp_size (Size): The input size of the model.
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
        """Constructor for the PPYOLOE class.

        Args:
            a_name (str): The name of the model.
            a_mdl_path (str): The path to the model file.
            a_mdl_device (str): The device on which the model will run.
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
        self.inp_size: Size = Size(640, 640)

    def _preproc(self, a_image: Image2D) -> np.ndarray:
        """Pre-process the input image before inference.

        Args:
            a_image (Image2D): The input image.

        Returns:
            np.ndarray: The pre-processed image in NumPy array format.
        """
        self.validate_mdl()

        # Resize Image
        image = cv2.resize(
            a_image.data,
            self.inp_size.to_tuple(),
            interpolation=cv2.INTER_AREA,
        )

        # Convert HWC to CHW array
        image = np.transpose(image, (2, 0, 1))

        # add batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def _postproc(self, a_preds: np.ndarray, a_image: Image2D) -> BBox2DList:
        """Post-process the predictions obtained from the model inference.

        Args:
            a_preds (np.ndarray): Predictions obtained from the model inference.
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: A list of detected persons after post-processing.
        """
        self.validate_mdl()

        persons: BBox2DList = BBox2DList()
        if a_preds is not None and len(a_preds):
            # Filter person class
            keep_idx = a_preds[:, 0] == 0
            preds = a_preds[keep_idx, :]

            # Apply Non-Maximum Suppression to filter out overlapping detections
            indices = cv2.dnn.NMSBoxes(
                bboxes=preds[:, 2:],
                scores=preds[:, 1],
                score_threshold=self.conf_thre,
                nms_threshold=self.nms_thre,
                top_k=self.top_k_thre,
            )
            nms_filtered_preds = preds[indices, :]
            nms_filtered_preds = nms_filtered_preds[:, [2, 3, 4, 5, 1]]

            # Scale Boxes Coordinates
            scaled_preds = nms_filtered_preds.copy()
            scaled_preds[:, [0, 2]] *= a_image.width
            scaled_preds[:, [1, 3]] *= a_image.height

            # Instantiate Persons
            persons = BBox2DList.from_xyxys(
                a_coordinates=scaled_preds,
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
        """Perform inference on the input image.

        Args:
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: A list of detected persons after inference and post-processing.
        """
        self.validate_mdl()

        # Pre-process input image
        proc_input = self._preproc(a_image=a_image)

        # Model inference
        preds = self.mdl(
            {
                "image": proc_input,
                "scale_factor": np.array([[self.inp_size.width, self.inp_size.width]]),
            }
        )[0]

        # Post-process predictions
        persons = self._postproc(a_preds=preds, a_image=a_image)

        return persons
