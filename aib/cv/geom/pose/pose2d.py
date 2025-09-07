"""Computer Vision - Geometry - Pose2D

This module provides an abstract base class for representing 2D poses defined by keypoints.
It includes a Pose2D class for representing a pose and a Pose2DList class for managing collections of poses.

Classes:
    Pose2D:
        An abstract base class for representing a 2D pose defined by keypoints.
    Pose2DList:
        A list-like container for Pose2D objects.

Type Variables:
    KPT: Type variable for keypoint types (KeyPoint2D).
    PST: Type variable for pose types (Pose2D).

Type Aliases:

"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional, TypeAlias, TypeVar, Union

from aib.cnt.b_list import BaseList
from aib.cv.geom.line2d import AnyLineKeyPoint2DList
from aib.cv.geom.point.kpoint2d import AnyKeyPoint2D, FloatKeyPoint2D, IntKeyPoint2D, KeyPoint2DList

if TYPE_CHECKING:
    AnyPose2D: TypeAlias = Union["Pose2D[IntKeyPoint2D]", "Pose2D[FloatKeyPoint2D]"]
    IntPose2D: TypeAlias = "Pose2D[IntKeyPoint2D]"
    FloatPose2D: TypeAlias = "Pose2D[FloatKeyPoint2D]"
    AnyPose2DList: TypeAlias = Union["Pose2DList[IntPose2D]", "Pose2DList[FloatPose2D]"]
    IntPose2DList: TypeAlias = "Pose2DList[IntPose2D]"
    FloatPose2DList: TypeAlias = "Pose2DList[FloatPose2D]"
else:
    AnyPose2D = Union["Pose2D[IntKeyPoint2D]", "Pose2D[FloatKeyPoint2D]"]
    IntPose2D = "Pose2D[IntKeyPoint2D]"
    FloatPose2D = "Pose2D[FloatKeyPoint2D]"
    AnyPose2DList = Union["Pose2DList[IntPose2D]", "Pose2DList[FloatPose2D]"]
    IntPose2DList = "Pose2DList[IntPose2D]"
    FloatPose2DList = "Pose2DList[FloatPose2D]"

KPT = TypeVar("KPT", bound=AnyKeyPoint2D, default=FloatKeyPoint2D)
PST = TypeVar("PST", bound=AnyPose2D, default=FloatPose2D)


class Pose2D(KeyPoint2DList[KPT], ABC):
    """Pose2D Abstract Base Class

    An abstract base class for representing a 2D pose defined by keypoints.
    It provides an interface for subclasses to implement specific pose types.

    Attributes:
        data (list[KeyPoint2D[KPT]]): A list of KeyPoint2D objects representing the pose.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[KPT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Pose2D",
    ):
        """Initialize a Pose2D.

        Args:
            a_iterable (Optional[Iterable[KPT]]): An optional iterable of KeyPoint2D objects.
            a_max_size (Optional[int]): The maximum size of the pose.
            a_name (str): The name of the pose.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @property
    @abstractmethod
    def limbs(self) -> AnyLineKeyPoint2DList:
        """Get the list of limbs as Line2D objects based on the pose's keypoints.

        Returns:
            AnyLineKeyPoint2DList: A list of Line2D objects representing the limbs.
        """
        raise NotImplementedError(
            "Subclasses must implement the limbs property to "
            "return a Line2DList of limbs based on the pose's keypoints."
        )


class Pose2DList(BaseList[PST]):
    """Pose2DList Class

    A list-like container for Pose2D objects. It extends BaseList to provide
    additional functionality specific to Pose2D objects.

    Attributes:
        data (list[Pose2D[PST]]): A list of Pose2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[PST]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Pose2DList",
    ):
        """Initialize a Pose2DList.

        Args:
            a_iterable (Optional[Iterable[PST]]): An optional iterable of Pose2D objects.
            a_max_size (Optional[int]): The maximum size of the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)


if not TYPE_CHECKING:
    IntPose2D = Pose2D[IntKeyPoint2D]
    FloatPose2D = Pose2D[FloatKeyPoint2D]
    IntPose2DList = Pose2DList[IntPose2D]
    FloatPose2DList = Pose2DList[FloatPose2D]
