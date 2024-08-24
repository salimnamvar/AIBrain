""" 2D Bounding Box Objects

    This module defines classes for working with 2D bounding boxes in a geometric space.
    The primary classes include:

    - Box2D: Represents a 2D bounding box defined by two corner points.
    - Box2DList: A container for a collection of Box2D objects.

    Bounding boxes support operations like intersection, union, and IoU (Intersection over Union).
"""

import math

# region Import Dependencies
from typing import Union, Tuple, List

import numpy as np

from brain.util.cv.shape import Size
from brain.util.cv.shape.pt import Point2D
from brain.util.obj import ExtBaseObject, BaseObjectList
from .fmt import cxyar_to_xyxy


# endregion Import Dependencies


class Box2D(ExtBaseObject):
    """Box-2D
    This module defines a 2D box object, `Box2D`, which represents a rectangular region in a 2D space. The box
    is defined by two corner points, `p1` and `p2`, both represented by instances of the `Point2D` class.

    Attributes:
        p1 (Point2D):
            The top-left point of the box as a :class:`Point2D`.
        p2 (Point2D):
            The bottom-right point of the box as a :class:`Point2D`.
        width (int):
            The width of the box.
        height (int):
            The height of the box.
        area (int):
            The area of the box.
        aspect_ratio (float):
            The aspect ratio of the box.
        size (Size):
            The size of the box in :class:`Size`.
        center (Point2D):
            The center point of the box as a :class:`Point2D`.
    """

    def __init__(self, a_p1: Point2D, a_p2: Point2D, a_name: str = "Box2D"):
        """Constructor for Box2D

        Args:
            a_p1 (Point2D):
                The first corner point of the box.
            a_p2 (Point2D):
                The second corner point of the box.
            a_name (str, optional):
                A string specifying the name of the box (default is 'Box2D').
        """
        super().__init__(a_name=a_name)
        self.p1: Point2D = a_p1
        self.p2: Point2D = a_p2

    def to_dict(self) -> dict:
        """Convert Box2D to a dictionary

        Returns:
            dict: A dictionary representing the :class:`Box2D` object.
        """
        dic = {"p1": self.p1.to_dict(), "p2": self.p2.to_dict()}
        return dic

    @property
    def p1(self) -> Point2D:
        """Getter for the top-left point (p1) of the box.

        Returns:
            Point2D: The top-left point (p1) of the box.
        """
        return self._p1

    @p1.setter
    def p1(self, a_point: Point2D):
        """Setter for the top-left point (p1) of the box.

        Args:
            a_point (Point2D): The new top-left point (p1) for the box.

        Raises:
            TypeError: If `a_point` is not an instance of :class:`Point2D`.
        """
        if a_point is None and not isinstance(a_point, Point2D):
            raise TypeError("The `a_point` should be a `Point2D`.")
        self._p1: Point2D = a_point

    @property
    def p2(self) -> Point2D:
        """Getter for the bottom-right point (p2) of the box.

        Returns:
            Point2D: The bottom-right point (p2) of the box.
        """
        return self._p2

    @p2.setter
    def p2(self, a_point: Point2D):
        """Setter for the bottom-right point (p2) of the box.

        Args:
            a_point (Point2D): The bottom-right point (p2) of the box.

        Raises:
            TypeError: If `a_point` is not an instance of :class:`Point2D`.
        """
        if a_point is None and not isinstance(a_point, Point2D):
            raise TypeError("The `a_point` should be a `Point2D`.")
        self._p2: Point2D = a_point

    @property
    def width(self) -> int:
        """Width of the box.

        Returns:
            int: The width of the box.
        """
        return int(self.p2.x - self.p1.x)

    @property
    def height(self) -> int:
        """Height of the box.

        Returns:
            int: The height of the box.
        """
        return int(self.p2.y - self.p1.y)

    @property
    def area(self) -> int:
        """Area of the box.

        Returns:
            int: The area of the box.
        """
        return self.size.area

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the box.

        Returns:
            int: The aspect ratio of the box.
        """
        return self.size.aspect_ratio

    def intersection(self, a_box2d: "Box2D") -> int:
        """Calculate the intersection area with another box.

        Args:
            a_box2d (Box2D): Another box for intersection calculation.

        Returns:
            int: The area of intersection between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        x_min, y_min = np.maximum((self.p1.x, self.p1.y), (a_box2d.p1.x, a_box2d.p1.y))
        x_max, y_max = np.minimum((self.p2.x, self.p2.y), (a_box2d.p2.x, a_box2d.p2.y))
        if x_min >= x_max or y_min >= y_max:
            return 0
        intersection_width = x_max - x_min
        intersection_height = y_max - y_min
        return int(intersection_width * intersection_height)

    def union(self, a_box2d: "Box2D") -> int:
        """Calculate the union area with another box.

        Args:
            a_box2d (Box2D): Another box for union calculation.

        Returns:
            int: The area of union between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return int(self.area + a_box2d.area - self.intersection(a_box2d))

    def iou(self, a_box2d: "Box2D") -> float:
        """Calculate the Intersection over Union (IoU) with another box.

        Args:
            a_box2d (Box2D): Another box for IoU calculation.

        Returns:
            float: The IoU between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return self.intersection(a_box2d) / self.union(a_box2d)

    def __add__(self, a_box2d: "Box2D") -> int:
        """Calculate the union area with another box.

        Args:
            a_box2d (Box2D): Another box for union area calculation.

        Returns:
            int: The union area between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return self.union(a_box2d)

    def __sub__(self, a_box2d: "Box2D") -> int:
        """Calculate the difference area with another box.

        Args:
            a_box2d (Box2D): Another box for difference area calculation.

        Returns:
            int: The difference area between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return int(self.area - self.intersection(a_box2d))

    def __mul__(self, a_box2d: "Box2D") -> int:
        """Calculate the intersection area with another box.

        Args:
            a_box2d (Box2D): Another box for intersection area calculation.

        Returns:
            int: The intersection area between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return self.intersection(a_box2d)

    def __truediv__(self, a_box2d: "Box2D") -> float:
        """Calculate the intersection over union (IoU) with another box.

        Args:
            a_box2d (Box2D): Another box for IoU calculation.

        Returns:
            float: The IoU between the current box and the provided box.
        Raises:
            TypeError: If `a_box2d` is not an instance of the `Box2D` class.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        return self.area / a_box2d.area

    def to_xyxy(self) -> np.ndarray:
        """Convert the box coordinates to a NumPy array [x, y, x, y].

        Returns:
            np.ndarray: A NumPy array containing the coordinates of both corner points.
        """
        return np.concatenate((self.p1.to_numpy(), self.p2.to_numpy()))

    @classmethod
    def validate_array(cls, a_coordinates: Union[Tuple, List, np.ndarray]) -> None:
        """Validate coordinates array

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding box in [x, y, x, y] format.

        Returns:
            This method does not return any values
        Raises:
            TypeError: If the input coordinates are not of type Tuple, List, or np.ndarray.
            ValueError: If the input coordinates do not have a length of 4 or have more than one dimension.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 4:
            raise ValueError(f"`a_coordinates` array should have length 4 but it is in shape of {a_coordinates.shape}.")

        if a_coordinates.ndim > 1:
            raise ValueError(f"`a_coordinates` array should be a 1D array but it has {a_coordinates.ndim} dimensions.")

    @classmethod
    def from_xyxy(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Box2D":
        """Create a Box2D instance from coordinates in xyxy format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the box in xyxy format.

            **kwargs:
                Additional keyword arguments to pass to the Box2D constructor.

        Returns:
            Box2D: An instance of the Box2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Instantiate box
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            **kwargs,
        )
        return box

    @property
    def size(self) -> Size:
        """Box2D's Size Getter

        This property specifies the size of the image in [Width, Height] format.

        Returns:
            Size:
                The size of the Box2D as :class:`Size`.
        """
        return Size(self.width, self.height, a_name=f"{self.name} Size")

    @property
    def center(self) -> Point2D:
        """Get the center point of the Box2D.

        Returns:
            Point2D: The center point of the box.
        """
        x = (self.p1.x + self.p2.x) / 2
        y = (self.p1.y + self.p2.y) / 2
        return Point2D(a_x=x, a_y=y, a_name=f"{self.name}'s Center Point")

    def __iter__(self):
        """Iterator for the Box object.

        Yields:
            Union[int, float]: The xyxy values in sequence.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y

    def __getitem__(self, index):
        """Get an element from the Box object by index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            Union[int, float]: The value at the specified index.

        Raises:
            IndexError: If the index is out of range for the Box object.
        """
        if index == 0:
            return self.p1.x
        elif index == 1:
            return self.p1.y
        elif index == 2:
            return self.p2.x
        elif index == 3:
            return self.p2.y
        else:
            raise IndexError("Index out of range for Box object")

    def to_cxyar(self) -> np.ndarray:
        """Convert the box coordinates to a NumPy array [center_x, center_y, area, aspect_ratio].

        Returns:
            np.ndarray: A NumPy array containing the coordinates of both corner points.
        """
        return np.concatenate((self.center.to_numpy(), [self.area], [self.aspect_ratio + 1e-6]))

    @classmethod
    def from_cxyar(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Box2D":
        """Create a Box2D instance from coordinates in (center-x, center-y, area, aspect_ratio) format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the box in (center-x, center-y, area, aspect_ratio) format.

            **kwargs:
                Additional keyword arguments to pass to the Box2D constructor.

        Returns:
            Box2D: An instance of the Box2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates)

        # Reformat (cx, cy, area, aspect_ratio) to (x, y, x, y)
        a_coordinates[:4] = cxyar_to_xyxy(*a_coordinates[:4])

        # Instantiate box
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            **kwargs,
        )
        return box

    def speed_xy(self, a_box2d: "Box2D") -> Tuple[float, float]:
        """Calculates the normalized speed vector from the center of the current bounding box to another bounding box in
        both x and y directions.

        Args:
            a_box2d (Box2D): The target bounding box to calculate the speed vector towards.

        Returns:
            Tuple[float, float]: A tuple representing the normalized speed vector (speed_x, speed_y).

        Raises:
            TypeError: If `a_box2d` is None or not an instance of Box2D.
        """
        if a_box2d is None or not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        speed_x, speed_y = self.center.speed_xy(a_point2d=a_box2d.center)
        return speed_x, speed_y

    def distance_centroid(self, a_box2d: "Box2D") -> float:
        """Calculate the Euclidean distance between the centroids of the current box and another box.

        Args:
            a_box2d (Box2D): Another box to calculate the distance to.

        Returns:
            float: The Euclidean distance between the centroids of the two boxes.

        Raises:
            TypeError: If `a_box2d` is None or not an instance of Box2D.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        distance = self.center.distance(a_point2d=a_box2d.center)
        return distance

    def distance_centroid_xy(self, a_box2d: "Box2D") -> Tuple[float, float]:
        """Calculate the distances in x and y directions between the centroids of the current box and another box.

        Args:
            a_box2d (Box2D): Another box to calculate the distances to.

        Returns:
            Tuple[float, float]: A tuple containing the distances in x and y directions between the centroids.

        Raises:
            TypeError: If `a_box2d` is None or not an instance of Box2D.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        distance_x, distance_y = self.center.distance_xy(a_point2d=a_box2d.center)
        return distance_x, distance_y

    def distance_min(self, a_box2d: "Box2D") -> float:
        """Calculate the minimum Euclidean distance between the edges of the current box and another box.

        Args:
            a_box2d (Box2D): Another box to calculate the minimum distance to.

        Returns:
            float: The minimum Euclidean distance between the edges of the two boxes.

        Raises:
            TypeError: If `a_box2d` is None or not an instance of Box2D.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        dx, dy = self.distance_min_xy(a_box2d=a_box2d)
        return math.sqrt(dx * dx + dy * dy)

    def distance_min_xy(self, a_box2d: "Box2D") -> Tuple[float, float]:
        """Calculate the minimum distances in x and y directions between the edges of the current box and another box.

        Args:
            a_box2d (Box2D): Another box to calculate the minimum distances to.

        Returns:
            Tuple[float, float]: A tuple containing the minimum distances in x and y directions between the edges.

        Raises:
            TypeError: If `a_box2d` is None or not an instance of Box2D.
        """
        if a_box2d is None and not isinstance(a_box2d, Box2D):
            raise TypeError("The `a_box2d` should be a `Box2D`.")
        dx = max(self.p1.x - a_box2d.p2.x, a_box2d.p1.x - self.p2.x, 0)
        dy = max(self.p1.y - a_box2d.p2.y, a_box2d.p1.y - self.p2.y, 0)
        return dx, dy


class Box2DList(BaseObjectList[Box2D]):
    """Box2D List

    The Box2DList class is based on the :class:`ObjectList` class and serves as a container for a collection of
    :class:`Box2D` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the Box2DList (default is 'Box2DList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[Box2D], optional):
            A list of Box2D objects to initialize the Box2DList (default is None).
    """

    def __init__(
        self,
        a_name: str = "Box2DList",
        a_max_size: int = -1,
        a_items: List[Box2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def from_xyxy(cls, a_coordinates: Union[Tuple, List, np.ndarray], **kwargs) -> "Box2DList":
        """Create a Box2DList instance from coordinates in xyxy format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the boxes in xyxy format.
            **kwargs:
                Additional keyword arguments to pass to the Box2D constructor.

        Returns:
            Box2DList: An instance of the Box2DList class.

        Raises:
            TypeError: If the input coordinates are not of type Tuple, List, or np.ndarray.
            ValueError: If the input coordinates do not have a length of 4 in the last dimension or have more than one
            dimension.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] != 4:
            raise ValueError(
                f"`a_coordinates` array should have length 4 in the last dimension(-1) but it has "
                f"{a_coordinates.shape}."
            )

        if a_coordinates.ndim == 1:
            a_coordinates = a_coordinates[np.newaxis]

        boxes = Box2DList()
        boxes.append(a_item=[Box2D.from_xyxy(coord, **kwargs) for coord in a_coordinates])
        return boxes

    def to_xyxy(self) -> np.ndarray:
        """Convert the Box2DList to a NumPy array [[x, y, x, y], ...].

        Returns:
            np.ndarray:
                A 2D NumPy array representing the concatenation of the `to_numpy` result for each Box2D object in the
                list.
        """
        return np.vstack([box.to_xyxy() for box in self.items])

    def to_cxyar(self) -> np.ndarray:
        """Convert the Box2DList to a NumPy array [[center_x, center_y, area, aspect_ratio], ...].

        Returns:
            np.ndarray:
                A 2D NumPy array representing the concatenation of the `to_numpy` result for each Box2D object in the
                list.
        """
        return np.vstack([box.to_cxyar() for box in self.items])
