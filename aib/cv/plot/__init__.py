"""Computer Vision - Plotting Utilities

Submodules:
    - collection: Plotting utilities for collection of geometry objects
    - geom: Plotting utilities for geometric objects
"""

# Import submodules
from . import collection, geom
from .collection import plot_geom, plot_geoms

# Flatten commonly used functions for convenience
from .geom import (
    plot_base_box2d,
    plot_base_point2d,
    plot_bbox2d,
    plot_box2d,
    plot_contour2d,
    plot_keypoint2d,
    plot_line2d,
    plot_point2d,
    plot_pose2d,
    plot_segbbox2d,
)

# Public API
__all__ = [
    # Submodules
    "geom",
    "collection",
    # Geometric plotting functions
    "plot_base_box2d",
    "plot_base_point2d",
    "plot_bbox2d",
    "plot_box2d",
    "plot_contour2d",
    "plot_keypoint2d",
    "plot_line2d",
    "plot_point2d",
    "plot_pose2d",
    "plot_segbbox2d",
    # General plotting functions
    "plot_geom",
    "plot_geoms",
]
