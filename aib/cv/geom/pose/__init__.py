"""Computer Vision - Geometry - Pose Utilities

Submodules:
    - pose2d: Base 2D pose utilities
    - coco_pose2d: COCO 2D pose utilities
"""

from typing import TYPE_CHECKING, TypeAlias, Union

# Import submodules
from . import coco_pose2d, pose2d

# Type aliases for easier type checking
if TYPE_CHECKING:
    AnyPose: TypeAlias = Union[pose2d.AnyPose2D, coco_pose2d.AnyCOCOPose2D]
    IntPose: TypeAlias = Union[pose2d.IntPose2D, coco_pose2d.IntCOCOPose2D]
    FloatPose: TypeAlias = Union[pose2d.FloatPose2D, coco_pose2d.FloatCOCOPose2D]

    AnyPoseList: TypeAlias = Union[pose2d.AnyPose2DList, coco_pose2d.AnyCOCOPose2DList]
    IntPoseList: TypeAlias = Union[pose2d.IntPose2DList, coco_pose2d.IntCOCOPose2DList]
    FloatPoseList: TypeAlias = Union[pose2d.FloatPose2DList, coco_pose2d.FloatCOCOPose2DList]
else:
    AnyPose = Union[pose2d.AnyPose2D, coco_pose2d.AnyCOCOPose2D]
    IntPose = Union[pose2d.IntPose2D, coco_pose2d.IntCOCOPose2D]
    FloatPose = Union[pose2d.FloatPose2D, coco_pose2d.FloatCOCOPose2D]

    AnyPoseList = Union[pose2d.AnyPose2DList, coco_pose2d.AnyCOCOPose2DList]
    IntPoseList = Union[pose2d.IntPose2DList, coco_pose2d.IntCOCOPose2DList]
    FloatPoseList = Union[pose2d.FloatPose2DList, coco_pose2d.FloatCOCOPose2DList]

# Public API
__all__ = [
    # Submodules
    "pose2d",
    "coco_pose2d",
    # Type aliases
    "AnyPose",
    "IntPose",
    "FloatPose",
    "AnyPoseList",
    "IntPoseList",
    "FloatPoseList",
    # COCO indices (expose from coco_pose2d)
    "COCO17KeyPointIndex",
    "COCO17LimbIndex",
]
