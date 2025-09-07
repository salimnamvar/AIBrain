"""Computer Vision - Geometry - Contour2D Utilities

This module provides utilities for working with 2D contours defined by a list of Point2D objects.
It includes a Contour2D class for representing a contour and a Contour2DList class for managing collections of contours.

Classes:
    Contour2D:
        Represents a 2D contour defined by a list of Point2D objects.
    Contour2DList:
        A list-like container for Contour2D objects with optional maximum size.

Type Variables:
    PT: Type variable for point types
    CT: Type variable for contour types

Type Aliases:
    AnyContour2D: Type alias for any contour type
    IntContour2D: Type alias for integer contour type
    FloatContour2D: Type alias for float contour type
"""

from typing import TYPE_CHECKING, Iterable, Optional, TypeAlias, TypeVar, Union

from scipy.spatial import KDTree

from src.utils.cv.geom.point.point2d import AnyPoint2D, FloatPoint2D, IntPoint2D, Point2DList, Point2DNestedList

if TYPE_CHECKING:
    AnyContour2D: TypeAlias = Union["Contour2D[IntPoint2D]", "Contour2D[FloatPoint2D]"]
    IntContour2D: TypeAlias = "Contour2D[IntPoint2D]"
    FloatContour2D: TypeAlias = "Contour2D[FloatPoint2D]"
    AnyContour2DList: TypeAlias = Union["Contour2DList[IntContour2D]", "Contour2DList[FloatContour2D]"]
    IntContour2DList: TypeAlias = "Contour2DList[IntContour2D]"
    FloatContour2DList: TypeAlias = "Contour2DList[FloatContour2D]"
else:
    AnyContour2D = Union["Contour2D[IntPoint2D]", "Contour2D[FloatPoint2D]"]
    IntContour2D = "Contour2D[IntPoint2D]"
    FloatContour2D = "Contour2D[FloatPoint2D]"
    AnyContour2DList = Union["Contour2DList[IntContour2D]", "Contour2DList[FloatContour2D]"]
    IntContour2DList = "Contour2DList[IntContour2D]"
    FloatContour2DList = "Contour2DList[FloatContour2D]"

PT = TypeVar("PT", bound=AnyPoint2D, default=FloatPoint2D)
CT = TypeVar("CT", bound=AnyContour2D, default=FloatContour2D)


class Contour2D(Point2DList[PT]):
    """Contour2D Data Class

    Represents a 2D contour defined by a list of Point2D objects.

    Attributes:
        data (List[PT]): List of Point2D objects representing the contour.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[PT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Contour2D",
    ):
        """Initialize a Contour2D.

        Args:
            a_iterable (Optional[Iterable[PT]]): An iterable of Point2D objects.
            a_max_size (Optional[int]): Maximum size of the contour.
            a_name (str): Name of the contour.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    def area(self) -> float:
        """Calculate the area of the contour using the shoelace formula.

        Returns:
            float: The area of the contour.
        """
        n = len(self.data)
        area = 0.0
        for i in range(n):
            x1, y1 = self.data[i]
            x2, y2 = self.data[(i + 1) % n]
            area += x1 * y2 - y1 * x2
        area = float(abs(area) / 2.0)
        return area

    def to_tree(self) -> KDTree:
        """Convert the contour to a KDTree for efficient spatial queries.

        Returns:
            KDTree: A KDTree representation of the contour's points.
        """
        return KDTree(self.to_xy())

    def tree_distance(self, a_contour2d: "Contour2D[PT]") -> float:
        """Calculate the minimum distance between this contour and another Contour2D.

        Args:
            a_contour2d (Contour2D[PT]): Another Contour2D object to compare with.

        Returns:
            float: The minimum distance between the two contours.
        """
        dis1, _ = self.to_tree().query(a_contour2d.to_xy())
        dis2, _ = a_contour2d.to_tree().query(self.to_xy())
        dis = min(dis1.min(), dis2.min())
        return float(dis)


class Contour2DList(Point2DNestedList[CT]):
    """Contour2DList Data Container Class

    A list of Contour2D objects, allowing for operations on multiple contours.
    Inherits from Point2DNestedList for handling nested structures.

    Attributes:
        data (List[CT]): List of Contour2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[CT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Contour2DList",
    ):
        """Initialize a Contour2DList.

        Args:
            a_iterable (Optional[Iterable[CT]]): An iterable of Contour2D objects.
            a_max_size (Optional[int]): Maximum size of the list.
            a_name (str): Name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)


if not TYPE_CHECKING:
    IntContour2D = Contour2D[IntPoint2D]
    FloatContour2D = Contour2D[FloatPoint2D]
    IntContour2DList = Contour2DList[IntContour2D]
    FloatContour2DList = Contour2DList[FloatContour2D]
