"""Computer Vision Utilities

Subpackages:
    - geom: Geometry-related utilities (boxes, points, lines, poses, sizes, etc.)
    - img: Image utilities
    - plot: Plotting utilities
    - vid: Video utilities
"""

# Subpackages
from . import geom, img, plot, vid

# Public API
__all__ = [
    # Subpackages
    "geom",
    "img",
    "plot",
    "vid",
]
