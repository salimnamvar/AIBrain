"""COCO Pose Modules

This module contains sub-utility modules for handling operations related to COCO-based Pose.

Classes:
    COCO17Keypoints: An enumeration of keypoint indices.
    Limb: A dataclass representing a limb with two keypoints.
    COCO17Limbs: An enumeration of limbs defined by pairs of keypoints.
    COCO17Pose2D: Class representing a 2D pose consisting of keypoints and limbs.
    COCO17Pose2DList: Class representing a list of Pose2D objects.
"""

# region Imported Dependencies
import dataclasses
from enum import Enum, IntEnum
from typing import List, Union

from aib.cv.shape.ps.limb import Limb2DList, Limb2D
from aib.cv.shape.pt import KeyPoint2DList, KeyPoint2D
from aib.obj import BaseObjectList


# endregion Imported Dependencies


class COCO17Keypoints(IntEnum):
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
        p1 (COCO17Keypoints): The first keypoint of the limb.
        p2 (COCO17Keypoints): The second keypoint of the limb.
    """

    p1: COCO17Keypoints
    p2: COCO17Keypoints


class COCO17Limbs(Enum):
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
        LIMB17 (Limb): Limb from LEFT_EAR to LEFT_SHOULDER.
        LIMB18 (Limb): Limb from RIGHT_EAR to RIGHT_SHOULDER.
    """

    LIMB1: Limb = Limb(COCO17Keypoints.RIGHT_EAR, COCO17Keypoints.RIGHT_EYE)
    LIMB2: Limb = Limb(COCO17Keypoints.RIGHT_EYE, COCO17Keypoints.NOSE)
    LIMB3: Limb = Limb(COCO17Keypoints.NOSE, COCO17Keypoints.LEFT_EYE)
    LIMB4: Limb = Limb(COCO17Keypoints.LEFT_EYE, COCO17Keypoints.LEFT_EAR)
    LIMB5: Limb = Limb(COCO17Keypoints.RIGHT_WRIST, COCO17Keypoints.RIGHT_ELBOW)
    LIMB6: Limb = Limb(COCO17Keypoints.RIGHT_ELBOW, COCO17Keypoints.RIGHT_SHOULDER)
    LIMB7: Limb = Limb(COCO17Keypoints.RIGHT_SHOULDER, COCO17Keypoints.LEFT_SHOULDER)
    LIMB8: Limb = Limb(COCO17Keypoints.LEFT_SHOULDER, COCO17Keypoints.LEFT_ELBOW)
    LIMB9: Limb = Limb(COCO17Keypoints.LEFT_ELBOW, COCO17Keypoints.LEFT_WRIST)
    LIMB10: Limb = Limb(COCO17Keypoints.RIGHT_SHOULDER, COCO17Keypoints.RIGHT_HIP)
    LIMB11: Limb = Limb(COCO17Keypoints.RIGHT_HIP, COCO17Keypoints.LEFT_HIP)
    LIMB12: Limb = Limb(COCO17Keypoints.LEFT_HIP, COCO17Keypoints.LEFT_SHOULDER)
    LIMB13: Limb = Limb(COCO17Keypoints.RIGHT_HIP, COCO17Keypoints.RIGHT_KNEE)
    LIMB14: Limb = Limb(COCO17Keypoints.RIGHT_KNEE, COCO17Keypoints.RIGHT_ANKLE)
    LIMB15: Limb = Limb(COCO17Keypoints.LEFT_HIP, COCO17Keypoints.LEFT_KNEE)
    LIMB16: Limb = Limb(COCO17Keypoints.LEFT_KNEE, COCO17Keypoints.LEFT_ANKLE)
    LIMB17: Limb = Limb(COCO17Keypoints.LEFT_EAR, COCO17Keypoints.LEFT_SHOULDER)
    LIMB18: Limb = Limb(COCO17Keypoints.RIGHT_EAR, COCO17Keypoints.RIGHT_SHOULDER)


class COCO17Pose2D(KeyPoint2DList, BaseObjectList[KeyPoint2D]):
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
        for limb in COCO17Limbs.__members__.values():
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
        return self.items[COCO17Keypoints.NOSE]

    @property
    def left_eye(self) -> KeyPoint2D:
        """Property to get the left eye keypoint.

        Returns:
            KeyPoint2D: Left eye keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_EYE]

    @property
    def right_eye(self) -> KeyPoint2D:
        """Property to get the right eye keypoint.

        Returns:
            KeyPoint2D: Right eye keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_EYE]

    @property
    def left_ear(self) -> KeyPoint2D:
        """Property to get the left ear keypoint.

        Returns:
            KeyPoint2D: Left ear keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_EAR]

    @property
    def right_ear(self) -> KeyPoint2D:
        """Property to get the right ear keypoint.

        Returns:
            KeyPoint2D: Right ear keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_EAR]

    @property
    def left_shoulder(self) -> KeyPoint2D:
        """Property to get the left shoulder keypoint.

        Returns:
            KeyPoint2D: Left shoulder keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> KeyPoint2D:
        """Property to get the right shoulder keypoint.

        Returns:
            KeyPoint2D: Right shoulder keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_SHOULDER]

    @property
    def left_elbow(self) -> KeyPoint2D:
        """Property to get the left elbow keypoint.

        Returns:
            KeyPoint2D: Left elbow keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_ELBOW]

    @property
    def right_elbow(self) -> KeyPoint2D:
        """Property to get the right elbow keypoint.

        Returns:
            KeyPoint2D: Right elbow keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_ELBOW]

    @property
    def left_wrist(self) -> KeyPoint2D:
        """Property to get the left wrist keypoint.

        Returns:
            KeyPoint2D: Left wrist keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_WRIST]

    @property
    def right_wrist(self) -> KeyPoint2D:
        """Property to get the right wrist keypoint.

        Returns:
            KeyPoint2D: Right wrist keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_WRIST]

    @property
    def left_hip(self) -> KeyPoint2D:
        """Property to get the left hip keypoint.

        Returns:
            KeyPoint2D: Left hip keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_HIP]

    @property
    def right_hip(self) -> KeyPoint2D:
        """Property to get the right hip keypoint.

        Returns:
            KeyPoint2D: Right hip keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_HIP]

    @property
    def left_knee(self) -> KeyPoint2D:
        """Property to get the left knee keypoint.

        Returns:
            KeyPoint2D: Left knee keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_KNEE]

    @property
    def right_knee(self) -> KeyPoint2D:
        """Property to get the right knee keypoint.

        Returns:
            KeyPoint2D: Right knee keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_KNEE]

    @property
    def left_ankle(self) -> KeyPoint2D:
        """Property to get the left ankle keypoint.

        Returns:
            KeyPoint2D: Left ankle keypoint.
        """
        return self.items[COCO17Keypoints.LEFT_ANKLE]

    @property
    def right_ankle(self) -> KeyPoint2D:
        """Property to get the right ankle keypoint.

        Returns:
            KeyPoint2D: Right ankle keypoint.
        """
        return self.items[COCO17Keypoints.RIGHT_ANKLE]


class COCO17Pose2DList(BaseObjectList[COCO17Pose2D]):
    """Class representing a list of Pose2D objects.

    Inherits from BaseObjectList[Pose2D].

    Attributes:
        name (str): Name of the Pose2DList object. Default is "Pose2DList".
        max_size (int): Maximum size of the Pose2D list. Default is -1.
        items (List[COCO17Pose2D]): List of Pose2D items.
    """

    def __init__(
        self,
        a_name: str = "COCO17Pose2DList",
        a_max_size: int = -1,
        a_items: Union[COCO17Pose2D, List[COCO17Pose2D], "COCO17Pose2DList"] = None,
    ):
        """
        Initializes a Pose2DList instance.

        Args:
            a_name (str, optional): The name of the list. Defaults to "Pose2DList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1 (no limit).
            a_items (List[Pose2D], optional): A list of Pose2D objects to initialize the list with. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
