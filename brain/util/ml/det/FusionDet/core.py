"""FusionDet Person Detector

This module defines the FusionDet class, which represents a person detector based on ensemble learning.
"""

# region Imported Dependencies
import threading
from typing import List

from brain.util.cv.img import Image2D
from brain.util.cv.shape import Size
from brain.util.cv.shape.bx import BBox2DNestedList, BBox2DList
from brain.util.ml.det import OVObjDetModel, OVObjDetModelList
from .postproc import FusionBox


# endregion Imported Dependencies


class FusionDet(OVObjDetModelList):
    """FusionDet Person Detector

    FusionDet is a person detector based on ensemble learning. It inherits from OVPrsDetModelList and BaseObjectList.

    Attributes:
        conf_thre (float): Confidence threshold for predictions.
        nms_thre (float): Non-maximum suppression threshold.
        top_k_thre (int): Top-k threshold for predictions.
        min_size_thre (Size): Minimum size threshold for bounding boxes.
        overlap_thre (float): Threshold for overlap between bounding boxes during aggregation.
        margin_thre (Size): Margin threshold for expanding bounding boxes during aggregation.
        fusion_box (FusionBox): FusionBox instance for post-processing aggregation.
    """

    def __init__(
        self,
        a_conf_thre: float,
        a_nms_thre: float,
        a_top_k_thre: int,
        a_min_size_thre: Size,
        a_margin_thre: Size,
        a_overlap_thre: float,
        a_name: str = "FusionDet",
        a_max_size: int = -1,
        a_items: List[OVObjDetModel] = None,
    ):
        """
        Constructor for the FusionDet class.

        Args:
            a_conf_thre (float): Confidence threshold for predictions.
            a_nms_thre (float): Non-maximum suppression threshold.
            a_top_k_thre (int): Top-k threshold for predictions.
            a_min_size_thre (Size): Minimum size threshold for bounding boxes.
            a_margin_thre (Size): Margin threshold for expanding bounding boxes during aggregation.
            a_overlap_thre (float): Threshold for overlap between bounding boxes during aggregation.
            a_name (str, optional): Name of the FusionDet instance (default is 'FusionDet').
            a_max_size (int, optional): Maximum size of the list (default is -1, indicating no size limit).
            a_items (List[OVObjDetModel], optional):
                List of OVObjDetModel objects to initialize the FusionDet (default is None).
        """
        super().__init__(
            a_conf_thre=a_conf_thre,
            a_nms_thre=a_nms_thre,
            a_top_k_thre=a_top_k_thre,
            a_min_size_thre=a_min_size_thre,
            a_margin_thre=a_margin_thre,
            a_overlap_thre=a_overlap_thre,
            a_name=a_name,
            a_max_size=a_max_size,
            a_items=a_items,
        )
        self.fusion_box: FusionBox = FusionBox(
            a_overlap_thre=self.overlap_thre,
            a_conf_thre=self.conf_thre,
            a_top_k_thre=self.top_k_thre,
            a_min_size_thre=self.min_size_thre,
            a_margin_thre=self.margin_thre,
        )

    def _postproc(self, a_persons: BBox2DNestedList, a_image: Image2D) -> BBox2DList:
        """Post-processes the detections from different detectors.

        Args:
            a_persons (BBox2DNestedList): List of person detections from different detectors.
            a_image (Image2D): Input image.

        Returns:
            BBox2DList: Combined and processed person detections.
        """
        persons = self.fusion_box.infer(a_persons=a_persons, a_image_size=a_image.size)
        return persons

    @staticmethod
    def detect(a_detector: OVObjDetModel, a_image: Image2D, a_dets: BBox2DNestedList):
        """Runs inference on a single detector.

        Args:
            a_detector (OVObjDetModel): Person detector model.
            a_image (Image2D): Input image.
            a_dets (BBox2DNestedList): List to store detection results.
        """
        detected_persons: BBox2DList = a_detector.infer(a_image=a_image)
        a_dets.append(detected_persons)

    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        """Runs inference on all detectors and combines results.

        Args:
            a_image (Image2D): Input image.

        Returns:
            BBox2DList: Combined person detections.
        """

        # Initialize a list for collecting detection results
        batch_persons = BBox2DNestedList()

        # Initialize a list to hold references to the thread objects
        threads = []

        # Start a separate thread for each person detector
        for detector in self.items:
            thread = threading.Thread(
                target=FusionDet.detect, args=(detector, a_image, batch_persons)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Post-thread detections
        persons = self._postproc(a_persons=batch_persons, a_image=a_image)

        return persons
