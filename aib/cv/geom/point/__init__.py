"""Computer Vision - Geometry - Point Utilities

Submodules:
    - point2d: 2D point utilities
    - kpoint2d: 2D Keypoint utilities
"""

from typing import TYPE_CHECKING, TypeAlias, Union

# Import submodules
from . import kpoint2d, point2d

# Type aliases for easier type checking
if TYPE_CHECKING:
    AnyPoint: TypeAlias = Union[point2d.AnyPoint2D, kpoint2d.AnyKeyPoint2D]
    IntPoint: TypeAlias = Union[point2d.IntPoint2D, kpoint2d.IntKeyPoint2D]
    FloatPoint: TypeAlias = Union[point2d.FloatPoint2D, kpoint2d.FloatKeyPoint2D]
    AnyPointList: TypeAlias = Union[point2d.AnyPoint2DList, kpoint2d.AnyKeyPoint2DList]
    IntPointList: TypeAlias = Union[point2d.IntPoint2DList, kpoint2d.IntKeyPoint2DList]
    FloatPointList: TypeAlias = Union[point2d.FloatPoint2DList, kpoint2d.FloatKeyPoint2DList]
    AnyPointNestedList: TypeAlias = Union[point2d.AnyPoint2DNestedList, kpoint2d.AnyKeyPoint2DNestedList]
    IntPointNestedList: TypeAlias = Union[point2d.IntPoint2DNestedList, kpoint2d.IntKeyPoint2DNestedList]
    FloatPointNestedList: TypeAlias = Union[point2d.FloatPoint2DNestedList, kpoint2d.FloatKeyPoint2DNestedList]
else:
    AnyPoint = Union[point2d.AnyPoint2D, kpoint2d.AnyKeyPoint2D]
    IntPoint = Union[point2d.IntPoint2D, kpoint2d.IntKeyPoint2D]
    FloatPoint = Union[point2d.FloatPoint2D, kpoint2d.FloatKeyPoint2D]
    AnyPointList = Union[point2d.AnyPoint2DList, kpoint2d.AnyKeyPoint2DList]
    IntPointList = Union[point2d.IntPoint2DList, kpoint2d.IntKeyPoint2DList]
    FloatPointList = Union[point2d.FloatPoint2DList, kpoint2d.FloatKeyPoint2DList]
    AnyPointNestedList = Union[point2d.AnyPoint2DNestedList, kpoint2d.AnyKeyPoint2DNestedList]
    IntPointNestedList = Union[point2d.IntPoint2DNestedList, kpoint2d.IntKeyPoint2DNestedList]
    FloatPointNestedList = Union[point2d.FloatPoint2DNestedList, kpoint2d.FloatKeyPoint2DNestedList]

# Public API
__all__ = [
    # Submodules
    "point2d",
    "kpoint2d",
    # Type aliases
    "AnyPoint",
    "IntPoint",
    "FloatPoint",
    "AnyPointList",
    "IntPointList",
    "FloatPointList",
    "AnyPointNestedList",
    "IntPointNestedList",
    "FloatPointNestedList",
]
