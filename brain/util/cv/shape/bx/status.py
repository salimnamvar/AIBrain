""" Bounding Box Statuses

    CoordStatus:
        Enum representing the coordination statuses of a bounding box.

    ConfidenceStatus:
        Enum representing the confidence statuses of a bounding box.

    SizeStatus:
        Enum representing the size statuses of a bounding box.
"""

# region Import Dependencies
from enum import Enum

# endregion Import Dependencies


class CoordStatus(Enum):
    """Coordination Status Enum

    This enum class, represents the possible coordination statuses of a bounding box. It is used to indicate
    whether a bounding box is within the bounds of an image or if it extends beyond the image boundaries.

    Enum Members:
        - UNKNOWN (int): Status indicating an unknown out-of-bound condition (-1).
        - VALID (int): Status indicating that the bounding box is within the image bounds and has correct
        coordinates (0).
        - PARTIALLY_OOB (int): Status indicating that the bounding box is partially out of bounds and extends beyond
        the image boundaries in one side (1).
        - ENTIRELY_OOB (int): Status indicating that the bounding box is entirely out of bounds and extends beyond
        the image boundaries in all sides (2).
        - INVALID_COORDINATES (int): Status indicating that the bounding box's coordinates are invalid (3).
    """

    UNKNOWN: int = -1
    VALID: int = 0
    PARTIALLY_OOB: int = 1
    ENTIRELY_OOB: int = 2
    INVALID_COORDINATES: int = 3


class ConfidenceStatus(Enum):
    """Confidence Status Enum

    This enum class represents the confidence statuses of a bounding box.

    Attributes:
        UNKNOWN (int): Indicates an unknown confidence status (-1).
        CONFIDENT (int): Indicates a confident bounding box (0).
        NOT_CONFIDENT (int): Indicates a not confident bounding box (1).
    """

    UNKNOWN: int = -1
    CONFIDENT: int = 0
    NOT_CONFIDENT: int = 1


class SizeStatus(Enum):
    """Size Status Enum

    This enum class represents the size statuses of a bounding box.

    Attributes:
        UNKNOWN (int): Indicates an unknown size status (-1).
        VALID (int): Indicates a valid sized bounding box (0).
        INVALID (int): Indicates an invalid sized bounding box (1).
    """

    UNKNOWN: int = -1
    VALID: int = 0
    INVALID: int = 1
