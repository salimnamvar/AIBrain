"""HTD0001 Text Detector

This module implements a Text Detector based on the OpenVino horizontal-text-detection-0001 model.
"""

# region Imported Dependencies
import cv2
import numpy as np
from brain.utils.cv.img import Image2D
from brain.utils.cv.shape import Size
from brain.utils.cv.shape.bx import CoordStatus, ConfidenceStatus, SizeStatus, BBox2DList
from brain.utils.ml.det.util import OVObjDetModel


# endregion Imported Dependencies


class HTD0001(OVObjDetModel):
    """Implements a Text Detector based on the OpenVino horizontal-text-detection-0001 model.

    This class inherits from OVTextDetModel and implements a Text Detector based on the OpenVino
    horizontal-text-detection-0001 model.

    Attributes:
        Inherits attributes from :class:`OVTextDetModel`.
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
    ):
        """Initialize the HTD0001 Text Detector.

        Args:
            a_name (str): The name of the text detector.
            a_mdl_path (str): The path to the model file.
            a_mdl_device (str): The device on which the model will be loaded.
            a_conf_thre (float): The confidence threshold for detections.
            a_nms_thre (float): The non-maximum suppression threshold for filtering overlapping detections.
            a_top_k_thre (int): The maximum number of detections to keep after non-maximum suppression.
            a_min_size_thre (Size): The minimum size threshold for filtering detections.
        """
        super().__init__(
            a_name,
            a_mdl_path,
            a_mdl_device,
            a_conf_thre,
            a_nms_thre,
            a_top_k_thre,
            a_min_size_thre,
        )

    def _preproc(self, a_image: Image2D) -> np.ndarray:
        """Pre-process the input image.

        Args:
            a_image (Image2D): The input image.

        Returns:
            np.ndarray: The pre-processed image.
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

        # add batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def _postproc(self, a_preds: np.ndarray, a_image: Image2D) -> BBox2DList:
        """Post-process the model predictions.

        Args:
            a_preds (np.ndarray): The model predictions.
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: The list of detected text boxes.
        """
        self.validate_mdl()

        text_boxes: BBox2DList = BBox2DList()
        if a_preds is not None and len(a_preds):
            # Apply Non-Maximum Suppression to filter out overlapping detections
            indices = cv2.dnn.NMSBoxes(
                bboxes=a_preds[:, 0:-1],
                scores=a_preds[:, -1],
                score_threshold=self.conf_thre,
                nms_threshold=self.nms_thre,
                top_k=self.top_k_thre,
            )
            nms_filtered_preds = a_preds[indices, :]

            # Normalization
            norm_preds = nms_filtered_preds.copy()
            norm_preds[:, [0, 2]] /= self.mdl_inp_size.width
            norm_preds[:, [1, 3]] /= self.mdl_inp_size.height

            # Scale Boxes Coordinates
            scaled_preds = norm_preds.copy()
            scaled_preds[:, [0, 2]] *= a_image.width
            scaled_preds[:, [1, 3]] *= a_image.height

            # Instantiate boxes
            text_boxes = BBox2DList.from_xyxys(
                a_coordinates=scaled_preds,
                a_img_size=a_image.size,
                a_conf_thre=self.conf_thre,
                a_min_size_thre=self.min_size_thre,
                a_do_validate=True,
            )

            # Remove invalid boxes
            text_boxes.remove(
                a_status=[
                    CoordStatus.ENTIRELY_OOB,
                    CoordStatus.INVALID_COORDINATES,
                    ConfidenceStatus.NOT_CONFIDENT,
                    SizeStatus.INVALID,
                ]
            )

            # Clamp bounding boxes
            text_boxes.clamp()

            # Remove invalid boxes
            text_boxes.remove(
                a_status=[
                    CoordStatus.INVALID_COORDINATES,
                    SizeStatus.INVALID,
                ]
            )
        return text_boxes

    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        """Perform inference on the input image.

        Args:
            a_image (Image2D): The input image.

        Returns:
            BBox2DList: The list of detected text boxes.

        """
        self.validate_mdl()

        # Pre-process input image
        proc_input = self._preproc(a_image=a_image)

        # Model inference
        preds = self.mdl.infer_new_request(proc_input)[0]

        # Post-process predictions
        boxes = self._postproc(a_preds=preds, a_image=a_image)

        return boxes
