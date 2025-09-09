"""Computer Vision - Video Utilities

Submodules:
    - cv_cap: OpenCV video utilities
    - de_cap: Decord video utilities
"""

# Import submodules
from . import cv_cap

# Flatten commonly used classes/functions for convenience
from .cv_cap import OpenCVVideoCapture

# Public API
__all__ = [
    # Submodules
    "cv_cap",
    # Commonly used classes/functions
    "OpenCVVideoCapture",
]
