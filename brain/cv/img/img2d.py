"""2D Image Module

    This Python file contains the definition of the Image2D class, which represents a 2D image in the format of
    [Height, Width, Channels].
"""

# region Imported Dependencies
from typing import Optional

import numpy as np

from brain.cv.shape import Size
from brain.misc import Time
from brain.obj import ExtBaseObject


# endregion Imported Dependencies


class Image2D(ExtBaseObject):
    """Image2D

    A class defining a 2D image in the format [Height, Width, Channels].

    Attributes:
        filename (str):
            The filename of the image.
        data (np.ndarray):
            The image data as a NumPy array.
        width (int):
            The image's width size.
        height (int):
            The image's height size.
        size:
            A :class:`Size` that specifies the size of the image in [Width, Height] format.
        channels:
            The number of channels in the image.
    """

    def __init__(
        self,
        a_data: np.ndarray,
        a_filename: Optional[str] = None,
        a_name: str = "IMAGE2D",
        a_time: Optional[Time] = None,
    ) -> None:
        """
        Constructor for the Image2D class.

        Args:
            a_data (np.ndarray):
                A NumPy array containing the image data.
            a_filename (Optional[str], optional):
                The filename of the image (default is None).
            a_name (str, optional):
                The name of the Image2D object (default is 'IMAGE2D').
            a_time (Time, optional):
                The time information related to the image.
        Returns:
            None:
                The constructor does not return any values.
        """

        super().__init__(a_name=a_name, a_time=a_time)
        self.data: np.ndarray = a_data
        self.filename: str = a_filename

    @property
    def data(self) -> np.ndarray:
        """Get the image data as a NumPy array."""
        return self._data

    @data.setter
    def data(self, a_data: np.ndarray) -> None:
        """Set the image data as a NumPy array.

        Args:
            a_data (np.ndarray):
                The image data to set.

        Raises:
            TypeError:
                If `a_data` is not an instance of `np.ndarray`.
            ValueError:
                If `a_data` array is not 2D or 3D with at least 1 channel.
        """
        if a_data is None or not isinstance(a_data, np.ndarray):
            raise TypeError(f"`a_data` argument must be an `np.ndarray` but it's type is `{type(a_data)}`")
        if a_data.ndim not in [2, 3]:
            raise ValueError(
                f"`a_data` array must be 2D or 3D with at least with 1 channel but it is in shape of "
                f"`{a_data.shape}`"
            )
        self._data: np.ndarray = a_data

    @property
    def filename(self) -> str:
        """Get the filename of the image."""
        return self._filename

    @filename.setter
    def filename(self, a_filename: Optional[str] = None) -> None:
        """Set the filename of the image.

        Args:
            a_filename (Optional[str], optional):
                The filename to set.

        Raises:
            TypeError:
                If `a_filename` is not an instance of `str`.
        """
        if a_filename is not None and not isinstance(a_filename, str):
            raise TypeError(f"`a_filename` argument must be an `str` but it's type is `{type(a_filename)}`")
        self._filename: str = a_filename

    @property
    def width(self) -> int:
        """Get the image's width size."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Get the image's height size."""
        return self.data.shape[0]

    @property
    def size(self) -> Size:
        """Get the size of the image in [Width, Height] format."""
        return Size(self.width, self.height, a_name=f"{self.name} Size")

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the image.

        Returns:
            int: The aspect ratio of the image.
        """
        return self.size.aspect_ratio

    @property
    def channels(self) -> int:
        """Get the number of channels in the image."""
        if self.data.ndim == 2:
            return 1
        else:
            return self.data.shape[2]

    def to_dict(self) -> dict:
        """Convert the Image2D object to a dictionary representation.

        Returns:
            dict:
                A dictionary representation of the Image2D object.
        """
        dic = {self.name: self.data}
        return dic
