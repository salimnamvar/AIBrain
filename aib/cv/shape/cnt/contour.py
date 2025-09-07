"""Contour2D Module

    This module defines a class for representing 2D contour and a container class for a collection of such contours.
"""

# region Import Dependencies
from typing import List

from scipy.spatial import KDTree

from aib.cv.shape.pt import Point2DList, Point2D, Point2DNestedList
from aib.obj import BaseObjectList

# endregion Import Dependencies


# TODO(doc): Complete the document of following class
class Contour2D(Point2DList, BaseObjectList[Point2D]):
    def __init__(
        self,
        a_items: List[Point2D] = None,
        a_name: str = "Contour2D",
        a_max_size: int = -1,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def area(self) -> float:
        n = len(self.items)
        area = 0.0
        for i in range(n):
            x1, y1 = self.items[i]
            x2, y2 = self.items[(i + 1) % n]
            area += x1 * y2 - y1 * x2
        area = abs(area) / 2.0
        return area

    @property
    def tree(self):
        return KDTree(self.to_xy())

    def distance_min(self, a_contour2d: "Contour2D") -> float:
        dis1, _ = self.tree.query(a_contour2d.to_xy())
        dis2, _ = a_contour2d.tree.query(self.to_xy())
        dis = min(dis1.min(), dis2.min())
        return dis


# TODO(doc): Complete the document of following class
class Contour2DList(Point2DNestedList, BaseObjectList[Contour2D]):
    def __init__(
        self,
        a_name: str = "Contour2DList",
        a_max_size: int = -1,
        a_items: List[Contour2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class Contour2DNestedList(BaseObjectList[Contour2DList]):
    def __init__(
        self,
        a_name: str = "Contour2DNestedList",
        a_max_size: int = -1,
        a_items: List[Contour2DList] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
