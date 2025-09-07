"""Computer Vision - Geometry - COCO17 Pose2D Utilities

This module provides utilities for working with 2D poses defined by COCO17 keypoints.
It includes a COCO17Pose2D class for representing a pose and a COCO17Pose2DList class for managing collections of poses.

Classes:
    COCO17KeyPointIndex: Enum for COCO17 keypoint indices.
    COCO17LimbIndex: Enum for COCO17 limb connections between keypoints.
    COCO17Pose2D: Represents a 2D pose defined by COCO17 keypoints.
    COCO17Pose2DList: A list-like container for COCO17Pose2D objects.

Type Variables:
    KPT (TypeVar):
        A generic type variable for keypoints, bound to AnyKeyPoint2D. Represents either IntKeyPoint2D or
        FloatKeyPoint2D. Default is FloatKeyPoint2D.

    CPT (TypeVar):
        A generic type variable for COCO poses, bound to AnyCOCOPose2D. Represents either COCO17Pose2D[IntKeyPoint2D]
        or COCO17Pose2D[FloatKeyPoint2D]. Default is FloatCOCOPose2D.

Type Aliases:
    AnyCOCOPose2D (TypeAlias):
        Union of COCO17Pose2D[IntKeyPoint2D] and COCO17Pose2D[FloatKeyPoint2D]. Represents any COCO17 pose with
        integer or float keypoints.

    IntCOCOPose2D (TypeAlias): COCO17Pose2D[IntKeyPoint2D]. Represents a COCO17 pose with integer keypoints.

    FloatCOCOPose2D (TypeAlias): COCO17Pose2D[FloatKeyPoint2D]. Represents a COCO17 pose with float keypoints.

    AnyCOCOPose2DList (TypeAlias):
        Union of COCO17Pose2DList[IntCOCOPose2D] and COCO17Pose2DList[FloatCOCOPose2D]. Represents a list of
        COCO17 poses with integer or float keypoints.

    IntCOCOPose2DList (TypeAlias):
        COCO17Pose2DList[IntCOCOPose2D]. Represents a list of COCO17 poses with integer keypoints.

    FloatCOCOPose2DList (TypeAlias):
        COCO17Pose2DList[FloatCOCOPose2D]. Represents a list of COCO17 poses with float keypoints.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Iterable, Optional, TypeAlias, TypeVar, Union, cast

from src.utils.cv.geom.line2d import FloatLineKeyPoint2D, FloatLineKeyPoint2DList
from src.utils.cv.geom.point.kpoint2d import AnyKeyPoint2D, FloatKeyPoint2D, IntKeyPoint2D
from src.utils.cv.geom.pose.pose2d import Pose2D, Pose2DList

if TYPE_CHECKING:
    AnyCOCOPose2D: TypeAlias = Union["COCO17Pose2D[IntKeyPoint2D]", "COCO17Pose2D[FloatKeyPoint2D]"]
    IntCOCOPose2D: TypeAlias = "COCO17Pose2D[IntKeyPoint2D]"
    FloatCOCOPose2D: TypeAlias = "COCO17Pose2D[FloatKeyPoint2D]"
    AnyCOCOPose2DList: TypeAlias = Union["COCO17Pose2DList[IntCOCOPose2D]", "COCO17Pose2DList[FloatCOCOPose2D]"]
    IntCOCOPose2DList: TypeAlias = "COCO17Pose2DList[IntCOCOPose2D]"
    FloatCOCOPose2DList: TypeAlias = "COCO17Pose2DList[FloatCOCOPose2D]"
else:
    AnyCOCOPose2D = Union["COCO17Pose2D[IntKeyPoint2D]", "COCO17Pose2D[FloatKeyPoint2D]"]
    IntCOCOPose2D = "COCO17Pose2D[IntKeyPoint2D]"
    FloatCOCOPose2D = "COCO17Pose2D[FloatKeyPoint2D]"
    AnyCOCOPose2DList = Union["COCO17Pose2DList[IntCOCOPose2D]", "COCO17Pose2DList[FloatCOCOPose2D]"]
    IntCOCOPose2DList = "COCO17Pose2DList[IntCOCOPose2D]"
    FloatCOCOPose2DList = "COCO17Pose2DList[FloatCOCOPose2D]"

KPT = TypeVar("KPT", bound=AnyKeyPoint2D, default=FloatKeyPoint2D)
CPT = TypeVar("CPT", bound=AnyCOCOPose2D, default=FloatCOCOPose2D)


class COCO17KeyPointIndex(IntEnum):
    """Enum for COCO17 keypoint indices."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class LimbIndex:
    """LimbIndex Data class

    Dataclass to represent a limb connection between two keypoints.

    Attributes:
        p1 (COCO17KeyPointIndex): The first keypoint index.
        p2 (COCO17KeyPointIndex): The second keypoint index.
    """

    p1: COCO17KeyPointIndex
    p2: COCO17KeyPointIndex


class COCO17LimbIndex(Enum):
    """COCO17LimbIndex Enum

    Enum to represent the limbs in COCO17 pose estimation.

    Attributes:
        LIMB1 to LIMB18: Each limb connects two keypoints.
    """

    LIMB1 = LimbIndex(COCO17KeyPointIndex.RIGHT_EAR, COCO17KeyPointIndex.RIGHT_EYE)
    LIMB2 = LimbIndex(COCO17KeyPointIndex.RIGHT_EYE, COCO17KeyPointIndex.NOSE)
    LIMB3 = LimbIndex(COCO17KeyPointIndex.NOSE, COCO17KeyPointIndex.LEFT_EYE)
    LIMB4 = LimbIndex(COCO17KeyPointIndex.LEFT_EYE, COCO17KeyPointIndex.LEFT_EAR)
    LIMB5 = LimbIndex(COCO17KeyPointIndex.RIGHT_WRIST, COCO17KeyPointIndex.RIGHT_ELBOW)
    LIMB6 = LimbIndex(COCO17KeyPointIndex.RIGHT_ELBOW, COCO17KeyPointIndex.RIGHT_SHOULDER)
    LIMB7 = LimbIndex(COCO17KeyPointIndex.RIGHT_SHOULDER, COCO17KeyPointIndex.LEFT_SHOULDER)
    LIMB8 = LimbIndex(COCO17KeyPointIndex.LEFT_SHOULDER, COCO17KeyPointIndex.LEFT_ELBOW)
    LIMB9 = LimbIndex(COCO17KeyPointIndex.LEFT_ELBOW, COCO17KeyPointIndex.LEFT_WRIST)
    LIMB10 = LimbIndex(COCO17KeyPointIndex.RIGHT_SHOULDER, COCO17KeyPointIndex.RIGHT_HIP)
    LIMB11 = LimbIndex(COCO17KeyPointIndex.RIGHT_HIP, COCO17KeyPointIndex.LEFT_HIP)
    LIMB12 = LimbIndex(COCO17KeyPointIndex.LEFT_HIP, COCO17KeyPointIndex.LEFT_SHOULDER)
    LIMB13 = LimbIndex(COCO17KeyPointIndex.RIGHT_HIP, COCO17KeyPointIndex.RIGHT_KNEE)
    LIMB14 = LimbIndex(COCO17KeyPointIndex.RIGHT_KNEE, COCO17KeyPointIndex.RIGHT_ANKLE)
    LIMB15 = LimbIndex(COCO17KeyPointIndex.LEFT_HIP, COCO17KeyPointIndex.LEFT_KNEE)
    LIMB16 = LimbIndex(COCO17KeyPointIndex.LEFT_KNEE, COCO17KeyPointIndex.LEFT_ANKLE)
    LIMB17 = LimbIndex(COCO17KeyPointIndex.LEFT_EAR, COCO17KeyPointIndex.LEFT_SHOULDER)
    LIMB18 = LimbIndex(COCO17KeyPointIndex.RIGHT_EAR, COCO17KeyPointIndex.RIGHT_SHOULDER)


class COCO17Pose2D(Pose2D[KPT]):
    """COCO17Pose2D Data Class

    Represents a pose in 2D space defined by COCO17 keypoints.

    Attributes:
        data (list[KeyPoint2D[KPT]]): A list of KeyPoint2D objects representing the pose.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[KPT]] = None,
        a_max_size: Optional[int] = 17,
        a_name: str = "COCO17Pose2D",
    ):
        """Initialize a COCO17Pose2D.

        Args:
            a_iterable (Optional[Iterable[KPT]]): An optional iterable of KeyPoint2D objects.
            a_max_size (Optional[int]): The maximum size of the pose, default is 17 for COCO17.
            a_name (str): The name of the pose.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @property
    def limbs(self) -> FloatLineKeyPoint2DList:
        """Get the list of limbs as Line2D objects based on COCO17 keypoints.

        Returns:
            Line2DList[KPT]: A list of Line2D objects representing the limbs.
        """
        return FloatLineKeyPoint2DList(
            [
                FloatLineKeyPoint2D(
                    p1=cast(FloatKeyPoint2D, self.data[limb.value.p1.value]),
                    p2=cast(FloatKeyPoint2D, self.data[limb.value.p2.value]),
                )
                for limb in COCO17LimbIndex.__members__.values()
            ]
        )

    @property
    def nose(self) -> KPT:
        """Get the nose keypoint.

        Returns:
            KPT: The nose keypoint.
        """
        return self.data[COCO17KeyPointIndex.NOSE]

    @property
    def left_eye(self) -> KPT:
        """Get the left eye keypoint.

        Returns:
            KPT: The left eye keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_EYE]

    @property
    def right_eye(self) -> KPT:
        """Get the right eye keypoint.

        Returns:
            KPT: The right eye keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_EYE]

    @property
    def left_ear(self) -> KPT:
        """Get the left ear keypoint.

        Returns:
            KPT: The left ear keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_EAR]

    @property
    def right_ear(self) -> KPT:
        """Get the right ear keypoint.

        Returns:
            KPT: The right ear keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_EAR]

    @property
    def left_shoulder(self) -> KPT:
        """Get the left shoulder keypoint.

        Returns:
            KPT: The left shoulder keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> KPT:
        """Get the right shoulder keypoint.

        Returns:
            KPT: The right shoulder keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_SHOULDER]

    @property
    def left_elbow(self) -> KPT:
        """Get the left elbow keypoint.

        Returns:
            KPT: The left elbow keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_ELBOW]

    @property
    def right_elbow(self) -> KPT:
        """Get the right elbow keypoint.

        Returns:
            KPT: The right elbow keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_ELBOW]

    @property
    def left_wrist(self) -> KPT:
        """Get the left wrist keypoint.

        Returns:
            KPT: The left wrist keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_WRIST]

    @property
    def right_wrist(self) -> KPT:
        """Get the right wrist keypoint.

        Returns:
            KPT: The right wrist keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_WRIST]

    @property
    def left_hip(self) -> KPT:
        """Get the left hip keypoint.

        Returns:
            KPT: The left hip keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_HIP]

    @property
    def right_hip(self) -> KPT:
        """Get the right hip keypoint.

        Returns:
            KPT: The right hip keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_HIP]

    @property
    def left_knee(self) -> KPT:
        """Get the left knee keypoint.

        Returns:
            KPT: The left knee keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_KNEE]

    @property
    def right_knee(self) -> KPT:
        """Get the right knee keypoint.

        Returns:
            KPT: The right knee keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_KNEE]

    @property
    def left_ankle(self) -> KPT:
        """Get the left ankle keypoint.

        Returns:
            KPT: The left ankle keypoint.
        """
        return self.data[COCO17KeyPointIndex.LEFT_ANKLE]

    @property
    def right_ankle(self) -> KPT:
        """Get the right ankle keypoint.

        Returns:
            KPT: The right ankle keypoint.
        """
        return self.data[COCO17KeyPointIndex.RIGHT_ANKLE]


class COCO17Pose2DList(Pose2DList[CPT]):
    """COCO17Pose2DList Data Container Class

    A list-like container for COCO17Pose2D objects, allowing for operations on collections of poses.

    Attributes:
        data (List[COCO17Pose2D[CPT]]): The list of COCO17Pose2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[CPT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "COCO17Pose2DList",
    ):
        """Initialize a COCO17Pose2DList.

        Args:
            a_iterable (Optional[Iterable[CPT]]): An optional iterable of COCO17Pose2D objects.
            a_max_size (Optional[int]): The maximum size of the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)


if not TYPE_CHECKING:
    IntCOCOPose2D = COCO17Pose2D[IntKeyPoint2D]
    FloatCOCOPose2D = COCO17Pose2D[FloatKeyPoint2D]
    IntCOCOPose2DList = COCO17Pose2DList[IntCOCOPose2D]
    FloatCOCOPose2DList = COCO17Pose2DList[FloatCOCOPose2D]
