"""Computer Vision - Image - Image Data Class Utilities

This module provides the Image2D data class, which represents a 2D image with associated metadata.

Classes:
    - Image2D: Represents a 2D image with metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy.typing as npt

from src.utils.cnt.b_data import BaseData
from src.utils.cv.geom.size import Size


@dataclass(frozen=True)
class Image2D(BaseData):
    """Image2D Data Class

    Represents a 2D image with associated metadata. It includes properties for width, height, size,
    aspect ratio, and number of channels.

    Attributes:
        data (npt.NDArray[Any]): The image data as a NumPy array.
        filename (Optional[str]): Optional filename associated with the image.
        width (int): Width of the image.
        height (int): Height of the image.
        size (Size): Size of the image.
        aspect_ratio (float): Aspect ratio of the image.
        channels (int): Number of channels in the image.
    """

    data: npt.NDArray[Any] = field(compare=False)
    filename: Optional[str] = field(default=None, compare=False)

    @property
    def width(self) -> int:
        """Width of the image.

        Returns:
            int: Width of the image.
        """
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Height of the image.

        Returns:
            int: Height of the image.
        """
        return self.data.shape[0]

    @property
    def size(self) -> Size[int]:
        """Size of the image.

        Returns:
            Size[int]: A Size object containing the width and height of the image.
        """
        return Size[int](width=int(self.width), height=int(self.height))

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the image.

        Returns:
            float: Aspect ratio of the image (width / height).
        """
        return self.size.aspect_ratio

    @property
    def channels(self) -> int:
        """Number of channels in the image.

        Returns:
            int: Number of channels in the image. Returns 1 for grayscale images.
        """
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    @property
    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert the image data to a NumPy array.

        Returns:
            npt.NDArray[Any]: The image data as a NumPy array.
        """
        return self.data
