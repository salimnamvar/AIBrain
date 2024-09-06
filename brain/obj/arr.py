""" Array2D

This module defines the Array2D class, which represents a 2D array with column headings.

Classes:
    Array2D (class): Represents a 2D array with column headings.
    Array2DDict (class): Dictionary to store Array2D instances with UUID keys.
"""

# region Imported Dependencies
import uuid
from typing import List, Union

import numpy as np

from brain.cv.shape.sz import Size
from brain.obj import ExtBaseObject, BaseObjectDict


# endregion Imported Dependencies


class Array2D(ExtBaseObject):
    """Array2D Class.

    A class representing a 2D array with associated column headings.

    Attributes:
        name (str): A string specifying the name of the Array2D instance.
        headings (List[str]): A list of strings representing column headings.
        data (np.ndarray): A NumPy array containing the data of the 2D array.
    """

    def __init__(self, a_data: np.ndarray, a_headings: List[str], a_name: str = "ARRAY2D") -> None:
        """Constructor for Array2D.

        Args:
            a_data (np.ndarray): A 2D NumPy array containing the data for the 2D array.
            a_headings (List[str]): A list of strings representing the column headings.
            a_name (str, optional): A string specifying the name of the Array2D instance (default is 'ARRAY2D').

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name)
        # Set data
        self.data: np.ndarray = a_data
        # Set headings
        self.headings: List[str] = a_headings

    @property
    def headings(self) -> List[str]:
        """Get or set the column headings.

        Returns:
            List[str]: A list of strings representing the column headings.
        """
        return self._headings

    @headings.setter
    def headings(self, a_headings: List[str]) -> None:
        """Get or set the column headings.

        Returns:
            List[str]: A list of strings representing the column headings.

        Raises:
            TypeError: If `a_headings` is not a list of strings.
            TypeError: If the number of headings does not match the number of columns in the array.
        """
        if (
            a_headings is None
            or not isinstance(a_headings, (list, tuple))
            or not all(isinstance(value, str) for value in a_headings)
        ):
            raise TypeError(f"`a_headings` argument must be an `list` of `str` values")

        if len(a_headings) != self.cols:
            raise TypeError(
                f"The number of heading titles in `a_headings` must be the same as number of columns in " f"the array"
            )
        self._headings = a_headings

    @property
    def data(self) -> np.ndarray:
        """Get or set the 2D array data.

        Returns:
            np.ndarray: A NumPy array representing the 2D data.

        Raises:
            TypeError: If `a_data` is not a NumPy array.
            ValueError: If `a_data` is not a 2D array.
        """
        return self._data

    @data.setter
    def data(self, a_data: np.ndarray) -> None:
        """Set the 2D array data.

        Args:
            a_data (np.ndarray): A NumPy array representing the 2D data.

        Raises:
            TypeError: If `a_data` is not a NumPy array.
            ValueError: If `a_data` is not a 2D array.
        """
        if a_data is None or not isinstance(a_data, np.ndarray):
            raise TypeError(f"`a_data` argument must be an `np.ndarray` but it's type is `{type(a_data)}`")
        if a_data.ndim != 2:
            raise ValueError(f"`a_data` array must be 2D, but it is in shape of `{a_data.shape}`")
        self._data: np.ndarray = a_data

    def to_dict(self) -> dict:
        """Convert the Array2D to a dictionary.

        Returns:
            dict: A dictionary representation of the Array2D, where the key is the name and the value is the data.
        """
        dic = {self.name: self.data}
        return dic

    @property
    def cols(self) -> int:
        """Get the number of columns in the Array2D.

        Returns:
            int: The number of columns.
        """
        return self.data.shape[1]

    @property
    def rows(self) -> int:
        """Get the number of rows in the Array2D.

        Returns:
            int: The number of rows.
        """
        return self.data.shape[0]

    @property
    def size(self) -> Size:
        """Get the size of the Array2D.

        Returns:
            Size: The size of the Array2D, represented as a Size object.
        """
        return Size(self.cols, self.rows, a_name=f"{self.name} Size")

    def __len__(self):
        """Get the number of rows in the Array2D.

        Returns:
            int: The number of rows in the Array2D.
        """
        return len(self.data)


class Array2DDict(BaseObjectDict[uuid.UUID, Array2D]):
    """Dictionary to store Array2D instances with UUID keys.

    This class extends the BaseObjectDict to store Array2D instances using UUID keys.

    Args:
        name (str, optional):
            The name of the Array2DDict.
        max_size (int, optional):
            Maximum size of the Array2DDict (default is -1, indicating no size limit).
        items (Dict[uuid.UUID, Array2D]):
            Dictionary containing Array2D objects with UUID keys.
    """

    def __init__(
        self,
        a_name: str = "Array2DDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[Array2D, List[Array2D]] = None,
    ):
        """Constructor for Array2DDict.

        Args:
            a_name (str, optional):
                The name of the Array2DDict (default is 'Array2DDict').
            a_max_size (int, optional):
                Maximum size of the Array2DDict (default is -1, indicating no size limit).
            a_key (Union[uuid.UUID, List[uuid.UUID]], optional):
                Key or list of keys to initialize the Array2DDict (default is None).
            a_value (Union[Array2D, List[Array2D]], optional):
                Value or list of values (Array2D instances) to initialize the Array2DDict (default is None).
        """
        super().__init__(a_name, a_max_size, a_key, a_value)
