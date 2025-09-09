"""Computer Vision - Geometry Utilities.

Submodules:
    - b_geom: Base geometry utilities
    - box: 2D Box utilities
    - contour2d: 2D Contour utilities
    - line2d: 2D Line utilities
    - point: 2D Point utilities
    - pose: 2D Pose utilities
    - size: 2D Size utilities
"""

# Submodules
from . import box, contour2d, line2d, point, pose, size

# Base geometry
from .b_geom import BaseGeom

# Public API
__all__ = [
    # Base
    "BaseGeom",
    # Submodules
    "box",
    "contour2d",
    "line2d",
    "point",
    "pose",
    "size",
]
