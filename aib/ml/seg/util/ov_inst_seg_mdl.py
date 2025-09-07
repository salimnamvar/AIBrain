"""OpenVino Instance Segmentation Base Model

This module defines the base class for instance segmentation models using the OpenVino library.

Classes:
    - OVInstSegModel: Base class for instance segmentation models using OpenVino.
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import List

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.ml.det.util import OVObjDetModel
from aib.obj import BaseObjectList
from .bbox import SegBBox2DList
from ...det import OVObjDetModelList


# endregion Imported Dependencies


class OVInstSegModel(OVObjDetModel):
    """OpenVino Instance Segmentation Model

    This class serves as the base for instance segmentation models that use the OpenVino library.

    Attributes:
        Inherits attributes from the parent class `OVObjDetModel`.
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
        Initializes a new instance of the OpenVino-based Instance Segmentation Model class.

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

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        """
        Pre-process input data.

        This method should be implemented by subclasses to define the pre-processing
        steps for the input data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        """
        Post-process predictions.

        This method should be implemented by subclasses to define the post-processing
        steps for the model predictions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        """
        Run inference on the model.

        This method must be implemented by subclasses.

        Args:
            *args: Variable-length arguments.
            a_image (Image2D): The input image.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            SegBBox2DList: A list of segmented bounding boxes detected in the image.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `infer`")


class OVInstSegModelList(OVObjDetModelList, BaseObjectList[OVInstSegModel], ABC):
    """OpenVino Instance Segmentation Model List

    The OVInstSegModelList class is based on the OVObjDetModelList class and serves as a container for a
    collection of OVInstSegModel objects.

    Attributes:
        conf_thre (float): Confidence threshold for predictions.
        nms_thre (float): Non-maximum suppression threshold.
        top_k_thre (int): Top-k threshold for predictions.
        min_size_thre (Size): Minimum size threshold for bounding boxes.
        overlap_thre (float): Threshold for overlap between bounding boxes during aggregation.
        margin_thre (Size): Margin threshold for expanding bounding boxes during aggregation.
    """

    def __init__(
        self,
        a_conf_thre: float,
        a_nms_thre: float,
        a_top_k_thre: int,
        a_min_size_thre: Size,
        a_margin_thre: Size,
        a_overlap_thre: float,
        a_name: str = "OVInstSegModelList",
        a_max_size: int = -1,
        a_items: List[OVInstSegModel] = None,
    ):
        """
        Initialize a new OVInstSegModelList instance.

        Args:
            a_conf_thre (float): Confidence threshold for predictions.
            a_nms_thre (float): Non-maximum suppression threshold.
            a_top_k_thre (int): Top-k threshold for predictions.
            a_min_size_thre (Size): Minimum size threshold for bounding boxes.
            a_margin_thre (Size): Margin threshold for expanding bounding boxes during aggregation.
            a_overlap_thre (float): Threshold for overlap between bounding boxes during aggregation.
            a_name (str, optional): Name of the OVInstSegModelList instance (default is 'OVInstSegModelList').
            a_max_size (int, optional): Maximum size of the list (default is -1, indicating no size limit).
            a_items (List[OVInstSegModel], optional):
                List of OVInstSegModel objects to initialize the list (default is None).
        """
        super().__init__(
            a_conf_thre,
            a_nms_thre,
            a_top_k_thre,
            a_min_size_thre,
            a_margin_thre,
            a_overlap_thre,
            a_name,
            a_max_size,
            a_items,
        )

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        """Infer method.

        An abstract method to be implemented by subclasses. It defines the process of making inferences
        based on the functionality.

        Args:
            *args: Variable-length arguments.
            a_image (Image2D): The input image.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            SegBBox2DList: A list of segmented bounding boxes detected in the image.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        NotImplementedError("Subclasses must implement `infer`")
