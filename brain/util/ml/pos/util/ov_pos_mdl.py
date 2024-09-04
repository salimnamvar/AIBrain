"""OpenVino Pose Estimation Model

This module provides classes for OpenVino-based pose estimation models.

Classes:
    OVSinglePoseEstimatorModel: An abstract base class for single-person pose estimation models.
    OVMultiPoseEstimatorModel: An abstract base class for multi-person pose estimation models.
"""

# region Import Dependencies
from abc import ABC, abstractmethod

from brain.util.cv.img import Image2D
from brain.util.cv.shape.bx import BBox2D
from brain.util.cv.shape.ps.coco_17 import COCO17Pose2DList, COCO17Pose2D
from brain.util.ml.util import OVModel


# endregion Import Dependencies


class OVSPosEstModel(OVModel, ABC):
    """Single Person Pose Estimator

    An abstract base class for single-person pose estimation models.

    Attributes:
        conf_thre (float): Confidence threshold for predictions.
    """

    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
        a_conf_thre: float,
    ):
        """Initializes an OVSinglePoseEstimatorModel object.

        Args:
            a_name (str): Name of the model.
            a_mdl_path (str): Path to the model file.
            a_mdl_device (str): Device to run the model on (e.g., CPU, GPU).
            a_conf_thre (float): Confidence threshold for detecting keypoints.
        """
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device)
        self.conf_thre: float = a_conf_thre

    # region Attributes
    @property
    def conf_thre(self) -> float:
        """float: Confidence threshold for detecting keypoints."""
        return self._conf_thre

    @conf_thre.setter
    def conf_thre(self, a_conf_thre: float):
        """Sets the confidence threshold for detecting keypoints.

        Args:
            a_conf_thre (float): Confidence threshold value.
        Raises:
            TypeError: If a_conf_thre is not a float.
        """
        if a_conf_thre is None or not isinstance(a_conf_thre, float):
            raise TypeError("The `a_conf_thre` should be a `float`.")
        self._conf_thre: float = a_conf_thre

    # endregion Attributes

    def to_dict(self) -> dict:
        """Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the `OVSinglePoseEstimatorModel`.
        """
        dic = {
            "name": self.name,
            "mdl_path": self.mdl_path,
            "mdl_device": self.mdl_device,
            "conf_thre": self.conf_thre,
        }
        return dic

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        """Preprocesses input data before inference."""
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        """Postprocesses inference results."""
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, a_image: Image2D, a_box: BBox2D) -> COCO17Pose2D:
        """Performs pose estimation on a single person.

        Args:
            a_image (Image2D): Input image.
            a_box (BBox2D): Bounding box containing the person.
        Returns:
            COCO17Pose2D: Estimated pose.
        """
        NotImplementedError("Subclasses must implement `infer`")


class OVMPosEstModel(OVModel, ABC):
    """Multi Person Pose Estimator

    An abstract base class for multi-person pose estimation models.

    Attributes:
        conf_thre (float): Confidence threshold for predictions.
    """

    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
        a_conf_thre: float,
    ):
        """Initializes an OVMultiPoseEstimatorModel object.

        Args:
            a_name (str): Name of the model.
            a_mdl_path (str): Path to the model file.
            a_mdl_device (str): Device to run the model on (e.g., CPU, GPU).
            a_conf_thre (float): Confidence threshold for detecting keypoints.
        """
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device)

        self.conf_thre: float = a_conf_thre

    # region Attributes
    @property
    def conf_thre(self) -> float:
        """float: Confidence threshold for detecting keypoints."""
        return self._conf_thre

    @conf_thre.setter
    def conf_thre(self, a_conf_thre: float):
        """Sets the confidence threshold for detecting keypoints.

        Args:
            a_conf_thre (float): Confidence threshold value.
        Raises:
            TypeError: If a_conf_thre is not a float.
        """
        if a_conf_thre is None or not isinstance(a_conf_thre, float):
            raise TypeError("The `a_conf_thre` should be a `float`.")
        self._conf_thre: float = a_conf_thre

    # endregion Attributes

    def to_dict(self) -> dict:
        """Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the `OVMultiPoseEstimatorModel`.
        """
        dic = {
            "name": self.name,
            "mdl_path": self.mdl_path,
            "mdl_device": self.mdl_device,
            "conf_thre": self.conf_thre,
        }
        return dic

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        """Preprocesses input data before inference."""
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        """Postprocesses inference results."""
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> COCO17Pose2DList:
        """Performs pose estimation on multiple people.

        Args:
            a_image (Image2D): Input image.
        Returns:
            COCO17Pose2DList: List of estimated poses.
        """
        NotImplementedError("Subclasses must implement `infer`")
