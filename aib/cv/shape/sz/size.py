""" Size Module

This module defines a `Size` class, which represents the dimensions of an object.

"""

# region Import Dependencies
import math
from typing import Union

import numpy as np

from aib.misc.type import is_int, is_float
from aib.obj import ExtBaseObject


# endregion Import Dependencies


class Size(ExtBaseObject):
    """Size Class

    The `Size` class represents the dimensions (width and height) of an object.

    Attributes:
        width (Union[int, float]):
            The width of the object.
        height (Union[int, float]):
            The height of the object.
        area (Union[int, float]):
            The area of the object.
        aspect_ratio (float):
            The aspect ratio of the object.
    """

    def __init__(
        self,
        a_width: Union[int, float],
        a_height: Union[int, float],
        a_name: str = "SIZE",
    ):
        """Constructor for the Size class.

        Args:
            a_width (Union[int, float]):
                The width of the object.
            a_height (Union[int, float]):
                The height of the object.
            a_name (str, optional):
                A string specifying the name of the Size object (default is 'SIZE').
        """
        super().__init__(a_name)
        self.width: Union[int, float] = a_width
        self.height: Union[int, float] = a_height

    def to_dict(self) -> dict:
        """Convert Size to a dictionary.

        Returns:
            dict: A dictionary representing the Size object with keys 'Width' and 'Height'.
        """
        dic: dict = {"Width": self.width, "Height": self.height}
        return dic

    @property
    def width(self) -> Union[int, float]:
        """Get the width of the Size.

        Returns:
            Union[int, float]: The width of the Size.
        """
        return self._width

    @width.setter
    def width(self, a_width: Union[int, float]):
        """Set the width of the Size.

        Args:
            a_width (Union[int, float]): The new width value.

        Raises:
            TypeError: If the provided width is not an integer or float.
        """
        int_flag = is_int(a_width)
        float_flag = is_float(a_width)
        if a_width is None and not (int_flag or float_flag):
            raise TypeError("The `a_width` should be a int or float.")
        if int_flag:
            a_width = int(a_width)
        if float_flag:
            a_width = float(a_width)
        self._width: Union[int, float] = a_width

    @property
    def height(self) -> Union[int, float]:
        """Get the height of the Size.

        Returns:
            Union[int, float]: The height value.
        """
        return self._height

    @height.setter
    def height(self, a_height: Union[int, float]):
        """Set the height of the Size.

        Args:
            a_height (Union[int, float]): The new height value.

        Raises:
            TypeError: If `a_height` is not an integer or float.
        """
        int_flag = is_int(a_height)
        float_flag = is_float(a_height)
        if a_height is None and not (int_flag or float_flag):
            raise TypeError("The `a_height` should be a int or float.")
        if int_flag:
            a_height = int(a_height)
        if float_flag:
            a_height = float(a_height)
        self._height: Union[int, float] = a_height

    def __iter__(self):
        """Iterator for the Size object.

        Yields:
            int: The width and height values in sequence.
        """
        yield self.width
        yield self.height

    def __getitem__(self, index):
        """Get an element from the Size object by index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            int: The value at the specified index.

        Raises:
            IndexError: If the index is out of range for the Size object.
        """
        if index == 0:
            return self.width
        elif index == 1:
            return self.height
        else:
            raise IndexError("Index out of range for Size object")

    def to_tuple(self) -> tuple:
        """Convert the Size object to a tuple.

        Returns:
            tuple: A tuple containing the width and height of the Size object.
        """
        return tuple(self)

    def __lt__(self, a_size: "Size"):
        """
        Compares the size with another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the current size is less than the given size, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return self.width < a_size.width and self.height < a_size.height

    def __le__(self, a_size: "Size"):
        """
        Compares the size with another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the current size is less than or equal to the given size, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return self.width <= a_size.width and self.height <= a_size.height

    def __eq__(self, a_size: "Size"):
        """
        Checks if the size is equal to another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the sizes are equal, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return self.width == a_size.width and self.height == a_size.height

    def __ne__(self, a_size: "Size") -> bool:
        """
        Checks if the size is not equal to another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the sizes are not equal, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None or not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return not (self.width == a_size.width and self.height == a_size.height)

    def __ge__(self, a_size: "Size"):
        """
        Compares the size with another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the current size is greater than or equal to the given size, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return self.width >= a_size.width and self.height >= a_size.height

    def __gt__(self, a_size: "Size"):
        """
        Compares the size with another Size object.

        Args:
            a_size (Size): The Size object to compare.

        Returns:
            bool: True if the current size is greater than the given size, False otherwise.

        Raises:
            TypeError: If `a_size` is not a valid Size object.
        """
        if a_size is None and not isinstance(a_size, Size):
            raise TypeError("The `a_size` should be a `Size`.")
        return self.width > a_size.width and self.height > a_size.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the Size.

        Returns:
            float: The aspect ratio, calculated as width / height.
        """
        return self.width / self.height

    @property
    def area(self) -> int | float:
        """Calculate the area of the Size.

        Returns:
            int: The area, calculated as width * height.
        """
        return self.width * self.height

    def hypot(self) -> float:
        """Calculate the diagonal length of the Size.

        This method uses the Pythagorean theorem to compute the diagonal length
        (or hypotenuse) of a rectangle defined by the width and height.

        Returns:
            float: The diagonal length of the Size, calculated as
                   sqrt(width^2 + height^2).
        """
        return math.hypot(self.width, self.height)

    def __mul__(self, a_size: Union[float, "Size"]) -> "Size":
        """Multiply Size by a float or another Size object.

        Args:
            a_size (Union[float, Size]): The multiplier, which can be a float or another Size object.

        Returns:
            Size: A new Size object with width and height multiplied.

        Raises:
            TypeError: If `other` is not a float or a valid Size object.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width * a_size, self.height * a_size)
        elif isinstance(a_size, Size):
            return Size(self.width * a_size.width, self.height * a_size.height)
        else:
            raise TypeError("The multiplier should be a float or a `Size`.")

    def __add__(self, a_size: Union[float, "Size"]) -> "Size":
        """Add a float or another Size object to the current Size.

        Args:
            a_size (Union[float, Size]): The value to add, which can be a float or another Size object.

        Returns:
            Size: A new Size object with width and height added.

        Raises:
            TypeError: If `other` is not a float or a valid Size object.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width + a_size, self.height + a_size)
        elif isinstance(a_size, Size):
            return Size(self.width + a_size.width, self.height + a_size.height)
        else:
            raise TypeError("The value to add should be a float or a `Size`.")

    def __sub__(self, a_size: Union[float, "Size"]) -> "Size":
        """Subtract a float or another Size object from the current Size.

        Args:
            a_size (Union[float, Size]): The value to subtract, which can be a float or another Size object.

        Returns:
            Size: A new Size object with width and height subtracted.

        Raises:
            TypeError: If `other` is not a float or a valid Size object.
        """
        if isinstance(a_size, (float, int)):
            return Size(self.width - a_size, self.height - a_size)
        elif isinstance(a_size, Size):
            return Size(self.width - a_size.width, self.height - a_size.height)
        else:
            raise TypeError("The value to subtract should be a float or a `Size`.")

    def __truediv__(self, a_size: Union[float, "Size"]) -> "Size":
        """Divide Size by a float or another Size object.

        Args:
            a_size (Union[float, Size]): The divisor, which can be a float or another Size object.

        Returns:
            Size: A new Size object with width and height divided.

        Raises:
            TypeError: If `other` is not a float or a valid Size object.
            ZeroDivisionError: If attempting to divide by zero.
        """
        if isinstance(a_size, (float, int)):
            if a_size == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return Size(self.width / a_size, self.height / a_size)
        elif isinstance(a_size, Size):
            if a_size.width == 0 or a_size.height == 0:
                raise ZeroDivisionError("Cannot divide by zero size dimensions.")
            return Size(self.width / a_size.width, self.height / a_size.height)
        else:
            raise TypeError("The divisor should be a float or a `Size`.")

    def to_list(self) -> list:
        """Convert to List

        Returns:
            list: A list representation of the size.
        """
        return list(self)

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy Array

        Returns:
            np.ndarray: A NumPy array representation of the size.
        """
        return np.asarray(self.to_list())
