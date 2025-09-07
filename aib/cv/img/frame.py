"""Computer Vision - Image - Frame Data Class Utilities

This module provides the Frame2D data class, which extends Image2D and BaseSeqData
to represent a 2D frame with additional metadata for video-specific data loading.

Classes:
    Frame2D: Represents a 2D frame with source metadata for video data handling.
"""

from dataclasses import dataclass, field
from os import PathLike
from typing import Optional

from aib.cnt.b_seq_data import BaseSeqData
from aib.cv.img.image import Image2D


@dataclass(frozen=True, kw_only=True)
class Frame2D(Image2D, BaseSeqData):
    """Frame2D Data Class

    Represents a 2D frame with additional metadata for video-specific data loading.
    Inherits from Image2D and BaseSeqData to provide image data and sequence data functionalities.

    Attributes:
        src_name (Optional[str]): Name of the source video, useful for identifying the video in multi-view setups.
        src_uri (Optional[str | int | PathLike[str]]):
            URI of the source video, useful for identifying the video in multi-view setups.
        src_id (Optional[int]): Identifier for the source of the frame, useful for multi-view data handling.
    """

    src_name: Optional[str] = field(default=None, compare=False)
    src_uri: Optional[str | int | PathLike[str]] = field(default=None, compare=False)
    src_id: Optional[int] = field(default=None, compare=False)
