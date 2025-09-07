"""Computer Vision - Geometry - Size Utilities

This module provides utilities for handling sizes in computer vision tasks.

Classes:
    - Size: Represents a size with width and height attributes, providing methods for arithmetic operations,
      comparisons, and conversions to NumPy arrays or tuples.

Type Variables:
    - _T: Type variable for numeric types (int or float).

Type Aliases:
    - AnySize: Type alias for a Size that can be either int or float.
    - IntSize: Type alias for a Size with integer dimensions.
    - FloatSize: Type alias for a Size with float dimensions.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt

from aib.cv.geom.b_geom import BaseGeom

_T = TypeVar("_T", bound=Union[int, float], default=float)

if TYPE_CHECKING:
    AnySize: TypeAlias = "Union[Size[int], Size[float]]"
    IntSize: TypeAlias = "Size[int]"
    FloatSize: TypeAlias = "Size[float]"
else:
    AnySize = "Union[Size[int], Size[float]]"
    IntSize = "Size[int]"
    FloatSize = "Size[float]"


@dataclass(frozen=True)
class Size(BaseGeom, Generic[_T]):
    """Size Data Class

    Represents a size with width and height attributes. Provides methods for arithmetic operations,
    comparisons, and conversions to NumPy arrays or tuples.

    Attributes:
        width (int | float): The width of the size.
        height (int | float): The height of the size.
        aspect_ratio (float): The aspect ratio of the size (width / height).
        area (int | float): The area of the size (width * height).
        hypot (float): The hypotenuse of the size dimensions (sqrt(width^2 + height^2)).
    """

    width: _T = field(compare=True)
    height: _T = field(compare=True)

    def __iter__(self):
        """Iterate over width and height.

        Yields:
            int | float: The width and height of the size.
        """
        yield self.width
        yield self.height

    def __getitem__(self, a_index: int):
        """Get the width or height based on the index.

        Args:
            a_index (int): The index to access (0 for width, 1 for height).

        Returns:
            int | float: The width if index is 0, height if index is 1.

        Raises:
            IndexError: If the index is not 0 or 1.
        """
        if a_index == 0:
            return self.width
        if a_index == 1:
            return self.height
        raise IndexError(f"Invalid index {a_index}: A `Size` object only supports indices 0 (width) and 1 (height).")

    def __lt__(self, a_size: AnySize) -> bool:
        """Check if the current Size is less than another Size."""
        return self.width < a_size.width and self.height < a_size.height

    def __le__(self, a_size: AnySize) -> bool:
        """Check if the current Size is less than or equal to another Size."""
        return self.width <= a_size.width and self.height <= a_size.height

    def __eq__(self, a_size: Any) -> bool:
        """Check if the current Size is equal to another Size."""
        if not isinstance(a_size, Size):
            return False
        return self.width == a_size.width and self.height == a_size.height

    def __ne__(self, a_size: object) -> bool:
        """Check if the current Size is not equal to another Size."""
        if not isinstance(a_size, Size):
            return True
        return not (self.width == a_size.width and self.height == a_size.height)

    def __ge__(self, a_size: AnySize) -> bool:
        """Check if the current Size is greater than or equal to another Size."""
        return self.width >= a_size.width and self.height >= a_size.height

    def __gt__(self, a_size: AnySize) -> bool:
        """Check if the current Size is greater than another Size."""
        return self.width > a_size.width and self.height > a_size.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the size.

        Returns:
            float: The aspect ratio calculated as width / height.
        """
        return self.width / float(self.height + 1e-6)

    @property
    def area(self) -> int | float:
        """Calculate the area of the size.

        Returns:
            int | float: The area calculated as width * height.
        """
        return self.width * self.height

    @property
    def hypot(self) -> float:
        """Calculate the hypotenuse of the size dimensions.

        Returns:
            float: The hypotenuse calculated as sqrt(width^2 + height^2).
        """
        return math.hypot(self.width, self.height)

    def __mul__(self, a_size: Union[float, int, AnySize]) -> AnySize:
        """Multiply Size by another Size.

        Args:
            a_size (Union[float, int, AnySize]): The scalar or Size to multiply by.

        Returns:
            AnySize: A new Size instance with the result of the multiplication.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width * a_size, self.height * a_size)
        return Size(self.width * a_size.width, self.height * a_size.height)

    def __add__(self, a_size: Union[float, int, AnySize]) -> AnySize:
        """Add Size to another Size or a scalar.

        Args:
            a_size (Union[float, int, AnySize]): The scalar or Size to add.

        Returns:
            AnySize: A new Size instance with the result of the addition.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width + a_size, self.height + a_size)
        return Size(self.width + a_size.width, self.height + a_size.height)

    def __sub__(self, a_size: Union[float, int, AnySize]) -> AnySize:
        """Subtract Size from another Size or a scalar.

        Args:
            a_size (Union[float, int, AnySize]): The scalar or Size to subtract.

        Returns:
            AnySize: A new Size instance with the result of the subtraction.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width - a_size, self.height - a_size)
        return Size(self.width - a_size.width, self.height - a_size.height)

    def __truediv__(self, a_size: Union[float, int, AnySize]) -> AnySize:
        """Divide Size by a scalar or another Size.

        Args:
            a_size (Union[float, int, AnySize]): The scalar or Size to divide by.

        Returns:
            AnySize: A new Size instance with the result of the division.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width / a_size, self.height / a_size)
        return Size(self.width / a_size.width, self.height / a_size.height)

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Convert Size to a NumPy array of type float32.

        Returns:
            npt.NDArray[np.float32]: A NumPy array with the width and height as float32.
        """
        return np.asarray(self.to_list(), dtype=np.float32)

    def to_int(self) -> IntSize:
        """Convert Size to a tuple of integers.

        Returns:
            IntSize: A Size instance with integer dimensions.
        """
        return Size[int](int(self.width), int(self.height))

    def to_float(self) -> FloatSize:
        """Convert Size to a tuple of floats.

        Returns:
            FloatSize: A Size instance with float dimensions.
        """
        return Size[float](float(self.width), float(self.height))

    @overload
    @classmethod
    def create(
        cls, a_width: Union[int, float], a_height: Union[int, float], a_use_float: Literal[True]
    ) -> FloatSize: ...

    @overload
    @classmethod
    def create(
        cls, a_width: Union[int, float], a_height: Union[int, float], a_use_float: Literal[False]
    ) -> IntSize: ...

    @overload
    @classmethod
    def create(cls, a_width: Union[int, float], a_height: Union[int, float], a_use_float: None = None) -> AnySize: ...

    @classmethod
    def create(
        cls, a_width: Union[int, float], a_height: Union[int, float], a_use_float: Optional[bool] = True
    ) -> AnySize:
        """Factory method to create a Size instance.

        Args:
            a_width (Union[int, float]): The width dimension.
            a_height (Union[int, float]): The height dimension.
            a_use_float (Optional[bool], optional): Whether to create a float Size.
                If True, creates Size[float]. If False, creates Size[int].
                Defaults to True. If None, determines based on input types.

        Returns:
            AnySize: A Size instance with either float or int dimensions.
        """
        if a_use_float is None:
            if all([isinstance(a_width, (int, np.integer)), isinstance(a_height, (int, np.integer))]):
                scalar_type = int
            else:
                scalar_type = float
        else:
            scalar_type = float if a_use_float else int
        return Size[scalar_type](scalar_type(a_width), scalar_type(a_height))


if not TYPE_CHECKING:
    IntSize = Size[int]
    FloatSize = Size[float]
