""" Size Module

This module defines a `Size` class, which represents the dimensions of an object.

"""

# region Import Dependencies
from typing import Union

from brain.util.misc import is_int, is_float
from brain.util.obj import BaseObject


# endregion Import Dependencies


class Size(BaseObject):
    """Size Class

    The `Size` class represents the dimensions (width and height) of an object.

    Attributes:
        width (Union[int, float]):
            The width of the object.
        height (Union[int, float]):
            The height of the object.
        area (int):
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
    def area(self) -> int:
        """Calculate the area of the Size.

        Returns:
            int: The area, calculated as width * height.
        """
        return self.width * self.height
