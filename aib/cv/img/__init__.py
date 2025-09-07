"""Computer Vision - Image Utilities

Submodules:
    - frame: Frame-related utilities
    - image: Image-related utilities
"""

# Import submodules
from . import frame, image

# commonly used classes for convenience
from .frame import Frame2D
from .image import Image2D

# Public API
__all__ = [
    # Submodules
    "frame",
    "image",
    # Commonly used classes
    "Frame2D",
    "Image2D",
]
