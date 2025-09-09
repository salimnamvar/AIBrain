"""Computer Vision - Geometry - Box Utilities

Submodules:
    - box2d: General Box Utilities
    - bbox2d: Bounding Box Utilities
    - sbbox2d: Segmented Bounding Box Utilities
    - utils: Conversion functions between coordinate formats
"""

from typing import TYPE_CHECKING, TypeAlias, Union

# Import Submodules
from . import bbox2d, box2d, sbbox2d, utils

# Flatten only key utility functions for convenience
from .utils import cxyar_to_xyxy, cxywh_to_xyxy, xywh_to_xyxy, xyxy_to_cxywh, xyxy_to_xywh

# Type aliases for easier type checking
if TYPE_CHECKING:
    AnyBox: TypeAlias = Union[box2d.AnyBox2D, bbox2d.AnyBBox2D, sbbox2d.AnySegBBox2D]
    IntBox: TypeAlias = Union[box2d.IntBox2D, bbox2d.IntBBox2D, sbbox2d.IntSegBBox2D]
    FloatBox: TypeAlias = Union[box2d.FloatBox2D, bbox2d.FloatBBox2D, sbbox2d.FloatSegBBox2D]
    AnyBoxList: TypeAlias = Union[box2d.AnyBox2DList, bbox2d.AnyBBox2DList, sbbox2d.AnySegBBox2DList]
    IntBoxList: TypeAlias = Union[box2d.IntBox2DList, bbox2d.IntBBox2DList, sbbox2d.IntSegBBox2DList]
    FloatBoxList: TypeAlias = Union[box2d.FloatBox2DList, bbox2d.FloatBBox2DList, sbbox2d.FloatSegBBox2DList]
    AnyBoxNestedList: TypeAlias = Union[
        box2d.AnyBox2DNestedList, bbox2d.AnyBBox2DNestedList, sbbox2d.AnySegBBox2DNestedList
    ]
    IntBoxNestedList: TypeAlias = Union[
        box2d.IntBox2DNestedList, bbox2d.IntBBox2DNestedList, sbbox2d.IntSegBBox2DNestedList
    ]
    FloatBoxNestedList: TypeAlias = Union[
        box2d.FloatBox2DNestedList, bbox2d.FloatBBox2DNestedList, sbbox2d.FloatSegBBox2DNestedList
    ]
else:
    AnyBox = Union[box2d.AnyBox2D, bbox2d.AnyBBox2D, sbbox2d.AnySegBBox2D]
    IntBox = Union[box2d.IntBox2D, bbox2d.IntBBox2D, sbbox2d.IntSegBBox2D]
    FloatBox = Union[box2d.FloatBox2D, bbox2d.FloatBBox2D, sbbox2d.FloatSegBBox2D]
    AnyBoxList = Union[box2d.AnyBox2DList, bbox2d.AnyBBox2DList, sbbox2d.AnySegBBox2DList]
    IntBoxList = Union[box2d.IntBox2DList, bbox2d.IntBBox2DList, sbbox2d.IntSegBBox2DList]
    FloatBoxList = Union[box2d.FloatBox2DList, bbox2d.FloatBBox2DList, sbbox2d.FloatSegBBox2DList]
    AnyBoxNestedList = Union[box2d.AnyBox2DNestedList, bbox2d.AnyBBox2DNestedList, sbbox2d.AnySegBBox2DNestedList]
    IntBoxNestedList = Union[box2d.IntBox2DNestedList, bbox2d.IntBBox2DNestedList, sbbox2d.IntSegBBox2DNestedList]
    FloatBoxNestedList = Union[
        box2d.FloatBox2DNestedList, bbox2d.FloatBBox2DNestedList, sbbox2d.FloatSegBBox2DNestedList
    ]

# Public API for convenience
__all__ = [
    # Submodules
    "bbox2d",
    "box2d",
    "sbbox2d",
    "utils",
    # Flattened utility functions
    "cxyar_to_xyxy",
    "cxywh_to_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_cxywh",
    "xyxy_to_xywh",
    # Type aliases
    "AnyBox",
    "IntBox",
    "FloatBox",
    "AnyBoxList",
    "IntBoxList",
    "FloatBoxList",
    "AnyBoxNestedList",
    "IntBoxNestedList",
    "FloatBoxNestedList",
]
