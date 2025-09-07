"""COCO Pose Modules

This module contains sub-utility modules for handling operations related to COCO-based Pose.

Classes:
    KeyPoint: An enumeration of keypoint indices.
    Limb: A dataclass representing a limb with two keypoints.
    Pose: An enumeration of limbs defined by pairs of keypoints.
    Pose2D: Class representing a 2D pose consisting of keypoints and limbs.
    Pose2DList: Class representing a list of Pose2D objects.
"""

# region Imported Dependencies
import dataclasses
from enum import Enum, IntEnum
from typing import List

from aib.cv.shape.ps.limb import Limb2DList, Limb2D
from aib.cv.shape.pt import KeyPoint2DList, KeyPoint2D
from aib.obj import BaseObjectList


# endregion Imported Dependencies


class KeyPoint(IntEnum):
    """Enumeration of keypoints used in pose estimation.

    Attributes:
        NOSE (int): Index of the nose keypoint.
        LEFT_EYE (int): Index of the left eye keypoint.
        RIGHT_EYE (int): Index of the right eye keypoint.
        LEFT_EAR (int): Index of the left ear keypoint.
        RIGHT_EAR (int): Index of the right ear keypoint.
        LEFT_SHOULDER (int): Index of the left shoulder keypoint.
        RIGHT_SHOULDER (int): Index of the right shoulder keypoint.
        LEFT_ELBOW (int): Index of the left elbow keypoint.
        RIGHT_ELBOW (int): Index of the right elbow keypoint.
        LEFT_WRIST (int): Index of the left wrist keypoint.
        RIGHT_WRIST (int): Index of the right wrist keypoint.
        LEFT_HIP (int): Index of the left hip keypoint.
        RIGHT_HIP (int): Index of the right hip keypoint.
        LEFT_KNEE (int): Index of the left knee keypoint.
        RIGHT_KNEE (int): Index of the right knee keypoint.
        LEFT_ANKLE (int): Index of the left ankle keypoint.
        RIGHT_ANKLE (int): Index of the right ankle keypoint.
    """

    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


@dataclasses.dataclass
class Limb:
    """A dataclass representing a limb with two keypoints.

    Attributes:
        p1 (KeyPoint): The first keypoint of the limb.
        p2 (KeyPoint): The second keypoint of the limb.
    """

    p1: KeyPoint
    p2: KeyPoint


class Pose(Enum):
    """Enumeration of limbs defined by pairs of keypoints for pose estimation.

    Attributes:
        LIMB1 (Limb): Limb from RIGHT_EAR to RIGHT_EYE.
        LIMB2 (Limb): Limb from RIGHT_EYE to NOSE.
        LIMB3 (Limb): Limb from NOSE to LEFT_EYE.
        LIMB4 (Limb): Limb from LEFT_EYE to LEFT_EAR.
        LIMB5 (Limb): Limb from RIGHT_WRIST to RIGHT_ELBOW.
        LIMB6 (Limb): Limb from RIGHT_ELBOW to RIGHT_SHOULDER.
        LIMB7 (Limb): Limb from RIGHT_SHOULDER to LEFT_SHOULDER.
        LIMB8 (Limb): Limb from LEFT_SHOULDER to LEFT_ELBOW.
        LIMB9 (Limb): Limb from LEFT_ELBOW to LEFT_WRIST.
        LIMB10 (Limb): Limb from RIGHT_SHOULDER to RIGHT_HIP.
        LIMB11 (Limb): Limb from RIGHT_HIP to LEFT_HIP.
        LIMB12 (Limb): Limb from LEFT_HIP to LEFT_SHOULDER.
        LIMB13 (Limb): Limb from RIGHT_HIP to RIGHT_KNEE.
        LIMB14 (Limb): Limb from RIGHT_KNEE to RIGHT_ANKLE.
        LIMB15 (Limb): Limb from LEFT_HIP to LEFT_KNEE.
        LIMB16 (Limb): Limb from LEFT_KNEE to LEFT_ANKLE.
    """

    LIMB1: Limb = Limb(KeyPoint.RIGHT_EAR, KeyPoint.RIGHT_EYE)
    LIMB2: Limb = Limb(KeyPoint.RIGHT_EYE, KeyPoint.NOSE)
    LIMB3: Limb = Limb(KeyPoint.NOSE, KeyPoint.LEFT_EYE)
    LIMB4: Limb = Limb(KeyPoint.LEFT_EYE, KeyPoint.LEFT_EAR)
    LIMB5: Limb = Limb(KeyPoint.RIGHT_WRIST, KeyPoint.RIGHT_ELBOW)
    LIMB6: Limb = Limb(KeyPoint.RIGHT_ELBOW, KeyPoint.RIGHT_SHOULDER)
    LIMB7: Limb = Limb(KeyPoint.RIGHT_SHOULDER, KeyPoint.LEFT_SHOULDER)
    LIMB8: Limb = Limb(KeyPoint.LEFT_SHOULDER, KeyPoint.LEFT_ELBOW)
    LIMB9: Limb = Limb(KeyPoint.LEFT_ELBOW, KeyPoint.LEFT_WRIST)
    LIMB10: Limb = Limb(KeyPoint.RIGHT_SHOULDER, KeyPoint.RIGHT_HIP)
    LIMB11: Limb = Limb(KeyPoint.RIGHT_HIP, KeyPoint.LEFT_HIP)
    LIMB12: Limb = Limb(KeyPoint.LEFT_HIP, KeyPoint.LEFT_SHOULDER)
    LIMB13: Limb = Limb(KeyPoint.RIGHT_HIP, KeyPoint.RIGHT_KNEE)
    LIMB14: Limb = Limb(KeyPoint.RIGHT_KNEE, KeyPoint.RIGHT_ANKLE)
    LIMB15: Limb = Limb(KeyPoint.LEFT_HIP, KeyPoint.LEFT_KNEE)
    LIMB16: Limb = Limb(KeyPoint.LEFT_KNEE, KeyPoint.LEFT_ANKLE)


class Pose2D(KeyPoint2DList, BaseObjectList[KeyPoint2D]):
    """Class representing a 2D pose consisting of keypoints and limbs.

    Inherits from KeyPoint2DList and BaseObjectList[KeyPoint2D].

    Attributes:
        name (str): Name of the Pose2D object. Default is "Pose2D".
        max_size (int): Maximum size of the keypoints list. Default is 17.
        items (List[KeyPoint2D]): List of KeyPoint2D items.
    """

    def __init__(
        self,
        a_name: str = "Pose2D",
        a_max_size: int = 17,
        a_items: List[KeyPoint2D] = None,
    ):
        """Initializes a Pose2D object.

        Args:
            a_name (str): Name of the Pose2D object. Default is "Pose2D".
            a_max_size (int): Maximum size of the keypoints list. Default is 17.
            a_items (List[KeyPoint2D], optional): List of KeyPoint2D items. Default is None.
        """
        super().__init__(a_name, a_max_size, a_items)

    @property
    def limbs(self) -> Limb2DList:
        """Property to get the list of limbs formed by the keypoints.

        Returns:
            Limb2DList: List of limbs.
        """
        limbs: Limb2DList = Limb2DList()
        for limb in Pose.__members__.values():
            limbs.append(
                a_item=Limb2D(
                    a_p1=self.items[limb.value.p1.value],
                    a_p2=self.items[limb.value.p2.value],
                    a_name=limb.value.p1.name + "-" + limb.value.p2.name,
                )
            )
        return limbs

    @property
    def nose(self) -> KeyPoint2D:
        """Property to get the nose keypoint.

        Returns:
            KeyPoint2D: Nose keypoint.
        """
        return self.items[KeyPoint.NOSE]

    @property
    def left_eye(self) -> KeyPoint2D:
        """Property to get the left eye keypoint.

        Returns:
            KeyPoint2D: Left eye keypoint.
        """
        return self.items[KeyPoint.LEFT_EYE]

    @property
    def right_eye(self) -> KeyPoint2D:
        """Property to get the right eye keypoint.

        Returns:
            KeyPoint2D: Right eye keypoint.
        """
        return self.items[KeyPoint.RIGHT_EYE]

    @property
    def left_ear(self) -> KeyPoint2D:
        """Property to get the left ear keypoint.

        Returns:
            KeyPoint2D: Left ear keypoint.
        """
        return self.items[KeyPoint.LEFT_EAR]

    @property
    def right_ear(self) -> KeyPoint2D:
        """Property to get the right ear keypoint.

        Returns:
            KeyPoint2D: Right ear keypoint.
        """
        return self.items[KeyPoint.RIGHT_EAR]

    @property
    def left_shoulder(self) -> KeyPoint2D:
        """Property to get the left shoulder keypoint.

        Returns:
            KeyPoint2D: Left shoulder keypoint.
        """
        return self.items[KeyPoint.LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> KeyPoint2D:
        """Property to get the right shoulder keypoint.

        Returns:
            KeyPoint2D: Right shoulder keypoint.
        """
        return self.items[KeyPoint.RIGHT_SHOULDER]

    @property
    def left_elbow(self) -> KeyPoint2D:
        """Property to get the left elbow keypoint.

        Returns:
            KeyPoint2D: Left elbow keypoint.
        """
        return self.items[KeyPoint.LEFT_ELBOW]

    @property
    def right_elbow(self) -> KeyPoint2D:
        """Property to get the right elbow keypoint.

        Returns:
            KeyPoint2D: Right elbow keypoint.
        """
        return self.items[KeyPoint.RIGHT_ELBOW]

    @property
    def left_wrist(self) -> KeyPoint2D:
        """Property to get the left wrist keypoint.

        Returns:
            KeyPoint2D: Left wrist keypoint.
        """
        return self.items[KeyPoint.LEFT_WRIST]

    @property
    def right_wrist(self) -> KeyPoint2D:
        """Property to get the right wrist keypoint.

        Returns:
            KeyPoint2D: Right wrist keypoint.
        """
        return self.items[KeyPoint.RIGHT_WRIST]

    @property
    def left_hip(self) -> KeyPoint2D:
        """Property to get the left hip keypoint.

        Returns:
            KeyPoint2D: Left hip keypoint.
        """
        return self.items[KeyPoint.LEFT_HIP]

    @property
    def right_hip(self) -> KeyPoint2D:
        """Property to get the right hip keypoint.

        Returns:
            KeyPoint2D: Right hip keypoint.
        """
        return self.items[KeyPoint.RIGHT_HIP]

    @property
    def left_knee(self) -> KeyPoint2D:
        """Property to get the left knee keypoint.

        Returns:
            KeyPoint2D: Left knee keypoint.
        """
        return self.items[KeyPoint.LEFT_KNEE]

    @property
    def right_knee(self) -> KeyPoint2D:
        """Property to get the right knee keypoint.

        Returns:
            KeyPoint2D: Right knee keypoint.
        """
        return self.items[KeyPoint.RIGHT_KNEE]

    @property
    def left_ankle(self) -> KeyPoint2D:
        """Property to get the left ankle keypoint.

        Returns:
            KeyPoint2D: Left ankle keypoint.
        """
        return self.items[KeyPoint.LEFT_ANKLE]

    @property
    def right_ankle(self) -> KeyPoint2D:
        """Property to get the right ankle keypoint.

        Returns:
            KeyPoint2D: Right ankle keypoint.
        """
        return self.items[KeyPoint.RIGHT_ANKLE]


class Pose2DList(BaseObjectList[Pose2D]):
    """Class representing a list of Pose2D objects.

    Inherits from BaseObjectList[Pose2D].

    Attributes:
        name (str): Name of the Pose2DList object. Default is "Pose2DList".
        max_size (int): Maximum size of the Pose2D list. Default is -1.
        items (List[Pose2D]): List of Pose2D items.
    """

    def __init__(
        self,
        a_name: str = "Pose2DList",
        a_max_size: int = -1,
        a_items: List[Pose2D] = None,
    ):
        """
        Initializes a Pose2DList instance.

        Args:
            a_name (str, optional): The name of the list. Defaults to "Pose2DList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1 (no limit).
            a_items (List[Pose2D], optional): A list of Pose2D objects to initialize the list with. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
