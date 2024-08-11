""" Point2D Module

    This module defines a class for representing 2D points and a container class for a collection of such points.
"""

# region Import Dependencies
from typing import Union, List, Tuple

import numpy as np

from brain.util.misc import is_int, is_float
from brain.util.obj import BaseObject, BaseObjectList


# endregion Import Dependencies


class Point2D(BaseObject):
    """Point2D

    This class represents a 2D point with x and y coordinates.

    Attributes:
        x (Union[int, float]):
            The x-coordinate of the point.
        y (Union[int, float]):
            The y-coordinate of the point.
    """

    def __init__(self, a_x: Union[int, float], a_y: Union[int, float], a_name: str = "POINT") -> None:
        """Constructor for Point2D

        Args:
            a_x (Union[int, float]):
                The x-coordinate value.
            a_y (Union[int, float]):
                The y-coordinate value.
            a_name (str, optional):
                A string specifying the name of the point (default is 'POINT').

        Raises:
            TypeError: If x and y values have different data types.
        """
        super().__init__(a_name)
        if type(a_x) != type(a_y):
            raise TypeError("X and Y values should both have the same data type")
        self.x = a_x
        self.y = a_y

    @property
    def x(self) -> Union[int, float]:
        """X Getter

        Returns:
            Union[int, float]: The x-coordinate of the point.
        """
        return self._x

    @x.setter
    def x(self, a_x: Union[int, float]) -> None:
        """X Setter

        Args:
            a_x (Union[int, float]):
                The new x-coordinate value.

        Raises:
            TypeError: If the provided x-coordinate is not an int or float.
        """
        int_flag = is_int(a_x)
        float_flag = is_float(a_x)
        if a_x is None and not (int_flag or float_flag):
            raise TypeError("The `a_x` should be a int or float.")
        if int_flag:
            a_x = int(a_x)
        if float_flag:
            a_x = float(a_x)
        self._x = a_x

    @property
    def y(self) -> Union[int, float]:
        """Y Getter

        Returns:
            Union[int, float]: The y-coordinate of the point.
        """
        return self._y

    @y.setter
    def y(self, a_y: Union[int, float]) -> None:
        """Y Setter

        Args:
            a_y (Union[int, float]):
                The new y-coordinate value.

        Raises:
            TypeError: If the provided y-coordinate is not an int or float.
        """
        int_flag = is_int(a_y)
        float_flag = is_float(a_y)
        if a_y is None and not (int_flag or float_flag):
            raise TypeError("The `a_y` should be a int or float.")
        if int_flag:
            a_y = int(a_y)
        if float_flag:
            a_y = float(a_y)
        self._y = a_y

    def to_dict(self) -> dict:
        """Convert to Dictionary

        Returns:
            dict: A dictionary representation of the point.
        """
        dic = {"x": self.x, "y": self.y}
        return dic

    def to_tuple(self) -> tuple:
        """Convert to Tuple

        Returns:
            tuple: A tuple representation of the point.
        """
        point = (self._x, self._y)
        return tuple(point)

    def to_list(self) -> list:
        """Convert to List

        Returns:
            list: A list representation of the point.
        """
        point = [self._x, self._y]
        return point

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy Array

        Returns:
            np.ndarray: A NumPy array representation of the point.
        """
        point = np.asarray(self.to_list())
        return point

    def to_xy(self) -> np.ndarray:
        """Convert to NumPy Array of format [x, y]

        Returns:
            np.ndarray: A NumPy array representation of the point.
        """
        return self.to_numpy()

    def to_int(self):
        """Converts the coordinates of the point to integers.

        This method modifies the `x` and `y` coordinates of the Point2D instance, rounding them to the nearest integer.
        """
        self._x = int(self._x)
        self._y = int(self._y)

    def to_float(self):
        """Converts the coordinates of the point to floats.

        This method modifies the `x` and `y` coordinates of the Point2D instance, converting them to floating-point
        numbers.
        """
        self._x = float(self._x)
        self._y = float(self._y)

    def __eq__(self, a_point2d: "Point2D") -> bool:
        """Check if two Point2D instances are equal.

        Args:
            a_point2d (Point2D): The Point2D instance to compare.

        Returns:
            bool: True if the coordinates of both Point2D instances are equal, False otherwise.

        Raises:
            TypeError: If `a_point2d` is not an instance of Point2D.
        """
        if a_point2d is None and not isinstance(a_point2d, Point2D):
            raise TypeError("The `a_point2d` should be a `Point2D`.")
        is_equal = self.x == a_point2d.x and self.y == a_point2d.y
        return is_equal

    def __iter__(self):
        """Iterator for the Point object.

        Yields:
            Union[int, float]: The x and y values in sequence.
        """
        yield self.x
        yield self.y

    def __getitem__(self, index):
        """Get an element from the Point object by index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            Union[int, float]: The value at the specified index.

        Raises:
            IndexError: If the index is out of range for the Point object.
        """
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range for Point object")

    def __hash__(self):
        """Hash Value

        The hash value is calculated based on the x and y coordinates of the point.

        Returns:
            int: The hash value of the Point2D object.
        """
        return hash((self.x, self.y))

    @classmethod
    def validate_array(cls, a_coordinates: Union[Tuple, List, np.ndarray]) -> np.ndarray:
        """Validate the array of coordinates.

        This method checks if the provided array of coordinates is a valid representation for creating keypoints.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                The array of coordinates to validate.

        Returns:
            np.ndarray: A validated and converted numpy array.

        Raises:
            TypeError:
                If `a_coordinates` is None or not an instance of Tuple, List, or np.ndarray.
            ValueError:
                If the array has less than 2 dimensions.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 2:
            raise ValueError(
                f"`a_coordinates` array should at least have length 2 but it is in shape of" f" {a_coordinates.shape}."
            )
        if a_coordinates.shape[0] == 1:
            a_coordinates = a_coordinates.flatten()
        return a_coordinates

    @classmethod
    def from_xy(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Point2D":
        """Create a Point2D instance from XY coordinates.

        This method creates a Point2D instance based on the provided XY coordinates and additional keyword arguments.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                The XY coordinates. For a simple point, pass a tuple or list with two elements (x, y).
            **kwargs:
                Additional keyword arguments to pass to the Point2D constructor.

        Returns:
            KeyPoint2D: A Point2D instance.

        Raises:
            ValueError:
                If the number of coordinates is less than 2.
        """
        coordinates = cls.validate_array(a_coordinates=a_coordinates)
        point = cls(a_x=coordinates[0], a_y=coordinates[1], **kwargs)
        return point

    def speed_xy(self, a_point2d: "Point2D") -> Tuple[float, float]:
        """Calculates the normalized speed vector from the current point to another point in both x and y directions.

        Args:
            a_point2d (Point2D): The target point to calculate the speed vector towards.

        Returns:
            tuple[float, float]: A tuple representing the normalized speed vector (speed_x, speed_y).

        Raises:
            TypeError: If `a_point` is None or not an instance of Point2D.
        """
        if a_point2d is None or not isinstance(a_point2d, Point2D):
            raise TypeError("The `a_point` should be a `Point2D`.")

        # Calculate the displacement vector between points
        speed = np.array([a_point2d.y - self.y, a_point2d.x - self.x])

        # Calculate the norm (magnitude) of the displacement vector
        norm = np.sqrt((a_point2d.y - self.y) ** 2 + (a_point2d.x - self.x) ** 2) + 1e-10

        # Normalize the speed vector
        normalized_speed = speed / norm

        # Extract components of the normalized speed vector
        speed_x = float(normalized_speed[0])
        speed_y = float(normalized_speed[1])
        return speed_x, speed_y

    def distance(self, a_point2d: "Point2D") -> float:
        """
        Calculate the Euclidean distance between the current point and another point.

        Args:
            a_point2d (Point2D): The target point to calculate the distance to.

        Returns:
            float: The Euclidean distance between the current point and the target point.

        Raises:
            TypeError: If `a_point2d` is None or not an instance of Point2D.
        """
        if a_point2d is None or not isinstance(a_point2d, Point2D):
            raise TypeError("The `a_point` should be a `Point2D`.")
        distance = np.linalg.norm(self.to_xy() - a_point2d.to_xy())
        return distance

    def distance_xy(self, a_point2d: "Point2D") -> Tuple[float, float]:
        """
        Calculate the absolute distances between the current point and another point along the x and y axes.

        Args:
            a_point2d (Point2D): The target point to calculate the distances to.

        Returns:
            tuple[float, float]:
                A tuple representing the absolute distances along the x and y axes (distance_x, distance_y).

        Raises:
            TypeError: If `a_point2d` is None or not an instance of Point2D.
        """
        if a_point2d is None or not isinstance(a_point2d, Point2D):
            raise TypeError("The `a_point` should be a `Point2D`.")
        distance_x = abs(self.x - a_point2d.x)
        distance_y = abs(self.y - a_point2d.y)
        return distance_x, distance_y


class Point2DList(BaseObjectList[Point2D]):
    """Point2D List

    The Point2DList class is based on the :class:`ObjectList` class and serves as a container for a collection of
    :class:`Point2D` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the Point2DList (default is 'Point2DList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[Point2D], optional):
            A list of Point2D objects to initialize the Point2DList (default is None).
    """

    def __init__(
        self,
        a_name: str = "Point2DList",
        a_max_size: int = -1,
        a_items: List[Point2D] = None,
    ):
        """
        Constructor for the Point2DList class.

        Args:
            name (str, optional):
                A string specifying the name of the Point2DList (default is 'Point2DList').
            max_size (int, optional):
                An integer representing the maximum size of the list (default is -1, indicating no size limit).
            items (List[Point2D], optional):
                A list of Point2D objects to initialize the Point2DList (default is None).

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def validate_array(cls, a_coordinates: Union[Tuple, List, np.ndarray]) -> np.ndarray:
        """Validate and convert the input array to a numpy array.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): The input array to be validated and converted.

        Returns:
            np.ndarray: A validated and converted numpy array.

        Raises:
            TypeError: If `a_coordinates` is None or not an instance of Tuple, List, or np.ndarray.
            ValueError: If the shape of the array is less than 2.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 2:
            raise ValueError(
                f"`a_coordinates` array should at least have length 2 but it is in shape of" f" {a_coordinates.shape}."
            )

        if a_coordinates.ndim == 1:
            a_coordinates = a_coordinates[np.newaxis]

        return a_coordinates

    def to_tuple(self) -> tuple:
        """Convert the `Point2DList` to a tuple.

        This method iterates through the `Point2D` objects in the `Point2DList` and converts them to a tuple.

        Returns:
            tuple: A tuple representation of the `Point2DList`.
        """
        points = ()
        for pnt in self.items:
            points += (pnt.to_tuple(),)
        return points

    def to_xy(self) -> np.ndarray:
        """Convert the Point2DList to a NumPy array of (x, y) coordinates.

        Returns:
            np.ndarray: A NumPy array containing (x, y) coordinates of all keypoints in the Point2DList.
        """
        return np.vstack([point.to_xy() for point in self.items])

    @classmethod
    def from_xy(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Point2DList":
        """Create a Point2DList from a list of (x, y) coordinates.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): The input coordinates to create Point2D instances.
            **kwargs: Additional keyword arguments to pass to the Point2D constructor.

        Returns:
            KeyPoint2DList: A Point2DList containing Point2D instances created from the input coordinates.

        Raises:
            TypeError: If `a_coordinates` is None or not an instance of Tuple, List, or np.ndarray.
            ValueError: If the shape of the array is less than 2.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate points
        points = cls()
        points.append(a_item=[Point2D.from_xy(a_coordinates=coord, **kwargs) for coord in coordinates])
        return points

    # TODO(doc): Complete the document of following method
    @property
    def center(self):
        pnt = np.mean(self.to_xy(), axis=0)
        return Point2D(a_x=pnt[0], a_y=pnt[1])


# TODO(doc): Complete the document of following class
class Point2DNestedList(BaseObjectList[Point2DList]):
    def __init__(
        self,
        a_name: str = "Point2DNestedList",
        a_max_size: int = -1,
        a_items: List[Point2DList] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def validate_array(cls, a_coordinates: Union[Tuple, List]) -> Union[Tuple, List]:
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List`.")
        return a_coordinates

    @classmethod
    def from_xy(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Point2DNestedList":
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate points
        points = cls()
        points.append(a_item=[Point2DList.from_xy(a_coordinates=coord, **kwargs) for coord in coordinates])
        return points
