"""Computer Vision - Geometry - Line2D Utilities

This module provides utilities for working with 2D lines defined by two points.
It includes a Line2D class for representing a line and a Line2DList class for managing collections of lines.

Classes:
    Line2D:
        Represents a line in 2D space defined by two points (p1, p2).
    Line2DList:
        A list-like container for Line2D objects with optional maximum size.

Type Variables:
    PT: Type variable for point types (Point2D or KeyPoint2D).
    LT: Type variable for line types (Line2D with point types).

Type Aliases:
    AnyLinePoint2D: Type alias for Line2D with any type of Point2D.
    IntLinePoint2D: Type alias for Line2D with integer type of Point2D.
    FloatLinePoint2D: Type alias for Line2D with float type of Point2D.
    AnyLineKeyPoint2D: Type alias for Line2D with any type of KeyPoint2D.
    IntLineKeyPoint2D: Type alias for Line2D with integer type of KeyPoint2D.
    FloatLineKeyPoint2D: Type alias for Line2D with float type of KeyPoint2D.
    AnyLine2D: Type alias for Line2D with any type of 2D points like Point2D or KeyPoint2D.
    FloatLine2D: Type alias for Line2D with float type of 2D points.
    IntLine2D: Type alias for Line2D with integer type of 2D points.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt

from src.utils.cnt.b_list import BaseList
from src.utils.cv.geom.b_geom import BaseGeom
from src.utils.cv.geom.point import AnyPoint, FloatPoint
from src.utils.cv.geom.point.kpoint2d import FloatKeyPoint2D, IntKeyPoint2D
from src.utils.cv.geom.point.point2d import FloatPoint2D, IntPoint2D

if TYPE_CHECKING:
    AnyLinePoint2D: TypeAlias = Union["Line2D[FloatPoint2D]", "Line2D[IntPoint2D]"]
    IntLinePoint2D: TypeAlias = "Line2D[IntPoint2D]"
    FloatLinePoint2D: TypeAlias = "Line2D[FloatPoint2D]"
    AnyLinePoint2DList: TypeAlias = Union["Line2DList[IntLinePoint2D]", "Line2DList[FloatLinePoint2D]"]
    IntLinePoint2DList: TypeAlias = "Line2DList[IntLinePoint2D]"
    FloatLinePoint2DList: TypeAlias = "Line2DList[FloatLinePoint2D]"
    AnyLineKeyPoint2D: TypeAlias = Union["Line2D[FloatKeyPoint2D]", "Line2D[IntKeyPoint2D]"]
    IntLineKeyPoint2D: TypeAlias = "Line2D[IntKeyPoint2D]"
    FloatLineKeyPoint2D: TypeAlias = "Line2D[FloatKeyPoint2D]"
    AnyLineKeyPoint2DList: TypeAlias = Union["Line2DList[IntLineKeyPoint2D]", "Line2DList[FloatLineKeyPoint2D]"]
    IntLineKeyPoint2DList: TypeAlias = "Line2DList[IntLineKeyPoint2D]"
    FloatLineKeyPoint2DList: TypeAlias = "Line2DList[FloatLineKeyPoint2D]"
    AnyLine2D: TypeAlias = Union[AnyLinePoint2D, AnyLineKeyPoint2D]
    FloatLine2D: TypeAlias = Union[FloatLinePoint2D, FloatLineKeyPoint2D]
    IntLine2D: TypeAlias = Union[IntLinePoint2D, IntLineKeyPoint2D]
    AnyLine2DList: TypeAlias = Union[
        "Line2DList[IntLineKeyPoint2D]",
        "Line2DList[FloatLineKeyPoint2D]",
        "Line2DList[IntLinePoint2D]",
        "Line2DList[FloatLinePoint2D]",
    ]
    IntLine2DList: TypeAlias = Union["Line2DList[IntLineKeyPoint2D]", "Line2DList[IntLinePoint2D]"]
    FloatLine2DList: TypeAlias = Union["Line2DList[FloatLineKeyPoint2D]", "Line2DList[FloatLinePoint2D]"]
else:
    AnyLinePoint2D = Union["Line2D[FloatPoint2D]", "Line2D[IntPoint2D]"]
    IntLinePoint2D = "Line2D[IntPoint2D]"
    FloatLinePoint2D = "Line2D[FloatPoint2D]"
    AnyLinePoint2DList = Union["Line2DList[IntLinePoint2D]", "Line2DList[FloatLinePoint2D]"]
    IntLinePoint2DList = "Line2DList[IntLinePoint2D]"
    FloatLinePoint2DList = "Line2DList[FloatLinePoint2D]"
    AnyLineKeyPoint2D = Union["Line2D[FloatKeyPoint2D]", "Line2D[IntKeyPoint2D]"]
    IntLineKeyPoint2D = "Line2D[IntKeyPoint2D]"
    FloatLineKeyPoint2D = "Line2D[FloatKeyPoint2D]"
    AnyLineKeyPoint2DList = Union["Line2DList[IntLineKeyPoint2D]", "Line2DList[FloatLineKeyPoint2D]"]
    IntLineKeyPoint2DList = "Line2DList[IntLineKeyPoint2D]"
    FloatLineKeyPoint2DList = "Line2DList[FloatLineKeyPoint2D]"
    AnyLine2D = Union[AnyLinePoint2D, AnyLineKeyPoint2D]
    FloatLine2D = Union[FloatLinePoint2D, FloatLineKeyPoint2D]
    IntLine2D = Union[IntLinePoint2D, IntLineKeyPoint2D]
    AnyLine2DList = Union[
        "Line2DList[IntLineKeyPoint2D]",
        "Line2DList[FloatLineKeyPoint2D]",
        "Line2DList[IntLinePoint2D]",
        "Line2DList[FloatLinePoint2D]",
    ]
    IntLine2DList = Union["Line2DList[IntLineKeyPoint2D]", "Line2DList[IntLinePoint2D]"]
    FloatLine2DList = Union["Line2DList[FloatLineKeyPoint2D]", "Line2DList[FloatLinePoint2D]"]


PT = TypeVar("PT", bound=AnyPoint, default=FloatPoint)
LT = TypeVar("LT", bound=AnyLine2D, default=FloatLine2D)


@dataclass(frozen=True)
class Line2D(BaseGeom, Generic[PT]):
    """Line2D Data Class

    Represents a line in 2D space defined by two points (p1, p2).

    Attributes:
        p1 (Point2D | KeyPoint2D): The first point of the line.
        p2 (Point2D | KeyPoint2D): The second point of the line.
    """

    p1: PT = field(compare=False)
    p2: PT = field(compare=False)

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the line to a 2D array of points.

        Returns:
            npt.NDArray[np.floating | np.integer]:
                A 2D array with shape (2, 2) containing the coordinates of the two points.
        """
        return np.vstack([self.p1.to_xy(), self.p2.to_xy()])


class Line2DList(BaseList[LT]):
    """Line2DList Data Container Class

    A list-like container for Line2D objects with optional maximum size.

    Attributes:
        data (list[LT]): The list of Line2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[LT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Line2DList",
    ):
        """Initialize a Line2DList.

        Args:
            a_iterable (Optional[Iterable[LT]]): An optional iterable of Line2D objects.
            a_max_size (Optional[int]): An optional maximum size for the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)


if not TYPE_CHECKING:
    IntLinePoint2D = Line2D[IntPoint2D]
    FloatLinePoint2D = Line2D[FloatPoint2D]
    IntLinePoint2DList = Line2DList[IntLinePoint2D]
    FloatLinePoint2DList = Line2DList[FloatLinePoint2D]
    IntLineKeyPoint2D = Line2D[IntKeyPoint2D]
    FloatLineKeyPoint2D = Line2D[FloatKeyPoint2D]
    IntLineKeyPoint2DList = Line2DList[IntLineKeyPoint2D]
    FloatLineKeyPoint2DList = Line2DList[FloatLineKeyPoint2D]
