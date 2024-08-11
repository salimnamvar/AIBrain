"""OpenVino Object Detector Model

This module defines utilities related to the OpenVino-based object detector model.

Classes:
    - OVObjDetModel: Represents an object detector model using OpenVino library.
    - OVObjDetModelList: Represents a container for a collection of OVObjDetModel objects.

"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import List

from brain.util.cv.img import Image2D
from brain.util.cv.shape import Size
from brain.util.cv.shape.bx import BBox2DList
from brain.util.misc import is_int
from brain.util.ml.util import OVModel, OVModelList
from brain.util.obj import BaseObjectList


# endregion Imported Dependencies


class OVObjDetModel(OVModel, ABC):
    """OpenVino Object Detector Model

    This class implements an object detector model using OpenVino library.

    Attributes:
        conf_thre (float): Confidence threshold for predictions.
        nms_thre (float): Non-maximum suppression threshold.
        top_k_thre (int): Top-k threshold for predictions.
        min_size_thre (Size): Minimum size threshold for bounding boxes.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
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
        """Constructor of Object Detector Model

        Initializes a new instance of the OpenVino-based Object Detector class.

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
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device)
        self.conf_thre: float = a_conf_thre
        self.nms_thre: float = a_nms_thre
        self.top_k_thre: int = a_top_k_thre
        self.min_size_thre: Size = a_min_size_thre
        self.classes: List[int] = a_classes

    # region Attributes
    @property
    def conf_thre(self) -> float:
        """Getter for the YOLOv8 confidence threshold.

        Returns:
            float: The confidence threshold used by the YOLOv8 model.
        """
        return self._conf_thre

    @conf_thre.setter
    def conf_thre(self, a_conf_thre: float):
        """Setter for the YOLOv8 confidence threshold.

        Args:
            a_conf_thre (float): The new confidence threshold.

        Raises:
            TypeError: If `a_conf_thre` is not a float.
        """
        if a_conf_thre is None or not isinstance(a_conf_thre, float):
            raise TypeError("The `a_conf_thre` should be a `float`.")
        self._conf_thre: float = a_conf_thre

    @property
    def nms_thre(self) -> float:
        """Getter for the YOLOv8 non-maximum suppression threshold.

        Returns:
            float: The non-maximum suppression threshold used by the YOLOv8 model.
        """
        return self._nms_thre

    @nms_thre.setter
    def nms_thre(self, a_nms_thre: float):
        """Setter for the YOLOv8 non-maximum suppression threshold.

        Args:
            a_nms_thre (float): The new non-maximum suppression threshold.

        Raises:
            TypeError: If `a_nms_thre` is not a float.
        """
        if a_nms_thre is None or not isinstance(a_nms_thre, float):
            raise TypeError("The `a_nms_thre` should be a `float`.")
        self._nms_thre: float = a_nms_thre

    @property
    def top_k_thre(self) -> int:
        """Getter for the YOLOv8 top-k threshold.

        Returns:
            int: The top-k threshold used by the YOLOv8 model.
        """
        return self._top_k_thre

    @top_k_thre.setter
    def top_k_thre(self, a_top_k_thre: int):
        """Setter for the YOLOv8 top-k threshold.

        Args:
            a_top_k_thre (int): The new top-k threshold.

        Raises:
            TypeError: If `a_top_k_thre` is not an int.
        """
        if a_top_k_thre is None or not isinstance(a_top_k_thre, int):
            raise TypeError("The `a_top_k_thre` should be a `int`.")
        self._top_k_thre: int = a_top_k_thre

    @property
    def min_size_thre(self) -> Size:
        """Getter for the YOLOv8 minimum size threshold.

        Returns:
            Size: The minimum size threshold used by the YOLOv8 model.
        """
        return self._min_size_thre

    @min_size_thre.setter
    def min_size_thre(self, a_min_size_thre: Size):
        """Setter for the YOLOv8 minimum size threshold.

        Args:
            a_min_size_thre (Size): The new minimum size threshold.

        Raises:
            TypeError: If `a_min_size_thre` is not a Size.
        """
        if a_min_size_thre is None or not isinstance(a_min_size_thre, Size):
            raise TypeError("The `a_min_size_thre` should be a `Size`.")
        self._min_size_thre: Size = a_min_size_thre

    @property
    def classes(self) -> List[int]:
        """
        Getter for the classes attribute.

        Returns:
            List[int]: The list of class indices to consider.
        """
        return self._classes

    @classes.setter
    def classes(self, a_classes: List[int]):
        """
        Setter for the classes attribute.

        Args:
            a_classes (List[int]): The new list of class indices.

        Raises:
            TypeError: If `a_classes` is not a list of int values.
        """
        if a_classes is not None and not isinstance(a_classes, list) and not is_int(a_classes):
            raise TypeError("The `a_classes` must be a List of int values.")
        self._classes: List[int] = a_classes

    # endregion Attributes

    def to_dict(self) -> dict:
        """Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the `OVObjDetModel`.
        """
        dic = {
            "name": self.name,
            "mdl_path": self.mdl_path,
            "mdl_device": self.mdl_device,
            "conf_thre": self.conf_thre,
            "nms_thre": self.nms_thre,
            "top_k_thre": self.top_k_thre,
            "min_size_thre": self.min_size_thre,
        }
        return dic

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        """Pre-process input data.

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
        """Post-process predictions.

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
    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        """Run inference on the model.

        This method must be implemented by subclasses.

        Args:
            *args: Variable-length arguments.
            a_image (Image2D): The input image.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            BBox2DList: A list of 2D bounding boxes detected in the image.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `infer`")


class OVObjDetModelList(OVModelList, BaseObjectList[OVObjDetModel], ABC):
    """OpenVino Object Detector Model List

    The OVObjDetModelList class is based on the :class:`BaseObjectList` class and serves as a container for a
    collection of :class:`OVObjDetModel` objects.

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
        a_name: str = "OVObjDetModelList",
        a_max_size: int = -1,
        a_items: List[OVObjDetModel] = None,
    ):
        """
        Constructor for the `OVObjDetModelList` class.

        Args:
            a_conf_thre (float): Confidence threshold for predictions.
            a_nms_thre (float): Non-maximum suppression threshold.
            a_top_k_thre (int): Top-k threshold for predictions.
            a_min_size_thre (Size): Minimum size threshold for bounding boxes.
            a_margin_thre (Size): Margin threshold for expanding bounding boxes during aggregation.
            a_overlap_thre (float): Threshold for overlap between bounding boxes during aggregation.
            a_name (str, optional): Name of the OVObjDetModelList instance (default is 'OVObjDetModelList').
            a_max_size (int, optional): Maximum size of the list (default is -1, indicating no size limit).
            a_items (List[OVObjDetModel], optional):
                List of OVObjDetModel objects to initialize the FusionDet (default is None).

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
        self.conf_thre: float = a_conf_thre
        self.nms_thre: float = a_nms_thre
        self.top_k_thre: int = a_top_k_thre
        self.min_size_thre: Size = a_min_size_thre
        self.overlap_thre: float = a_overlap_thre
        self.margin_thre: Size = a_margin_thre

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        """Inference Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of making inferences
        based n the functionality.

        Args:
            *args: Variable-length arguments.
            a_image (Image2D): The input image.
            **kwargs: Keyword arguments.

        Returns:
            BBox2DList: A list of 2D bounding boxes detected in the image.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `infer`")
