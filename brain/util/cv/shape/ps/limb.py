"""Limb2D Modules

This module contains utility classes for handling operations related to pose estimation.
"""

# region Imported Dependencies
from typing import List
import numpy as np
from brain.util.cv.shape.pt import KeyPoint2D
from brain.util.obj import ExtBaseObject, BaseObjectList

# endregion Imported Dependencies


class Limb2D(ExtBaseObject):
    """Represents a 2D limb consisting of two keypoints.

    Attributes:
        p1 (KeyPoint2D): The first keypoint of the limb.
        p2 (KeyPoint2D): The second keypoint of the limb.
    """

    def __init__(self, a_p1: KeyPoint2D, a_p2: KeyPoint2D, a_name: str = "Limb2D"):
        """
        Initializes a Limb2D instance.

        Args:
            a_p1 (KeyPoint2D): The first keypoint of the limb.
            a_p2 (KeyPoint2D): The second keypoint of the limb.
            a_name (str, optional): The name of the limb. Defaults to "Limb2D".
        """
        super().__init__(a_name)
        self.p1: KeyPoint2D = a_p1
        self.p2: KeyPoint2D = a_p2

    def to_dict(self) -> dict:
        """Converts the limb to a dictionary format.

        Returns:
            dict: A dictionary with keys 'P1' and 'P2' representing the keypoints of the limb.
        """
        dic = {"P1": self.p1, "P2": self.p2}
        return dic

    def to_xy(self) -> np.ndarray:
        """Converts the limb to an array of coordinates.

        Returns:
            np.ndarray: A 2x2 array where each row represents the (x, y) coordinates of the keypoints.
        """
        return np.vstack([self.p1.to_xy(), self.p2.to_xy()])


class Limb2DList(BaseObjectList[Limb2D]):
    """Represents a list of Limb2D objects.

    Attributes:
        name (str): The name of the list.
        max_size (int): The maximum size of the list.
        items (List[Limb2D]): The list of Limb2D objects.
    """

    def __init__(
        self,
        a_name: str = "Limb2DList",
        a_max_size: int = -1,
        a_items: List[Limb2D] = None,
    ):
        """
        Initializes a Limb2DList instance.

        Args:
            a_name (str, optional): The name of the list. Defaults to "Limb2DList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1 (no limit).
            a_items (List[Limb2D], optional): A list of Limb2D objects to initialize the list with. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
