"""Machine Learning Utilities

Subpackages:
    - det: Object Detection Utilities
"""

from . import det, trk
from .utils import BaseMLModel

__all__ = [
    # Subpackages
    "det",
    "trk",
    # Utilities
    "BaseMLModel",
]
