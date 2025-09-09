"""Computer Vision - Geometry - Point2D Utilities

This module provides utilities for working with 2D points, including a Point2D class, a list of Point2D objects, and
a nested list of Point2DList objects.

Classes:
    - Point2D: Represents a point in 2D space with x and y coordinates.
    - Point2DList: A list-like container for Point2D objects.
    - Point2DNestedList: A list-like container for lists of Point2D objects.

Type Variables:
    - T: Type variable for scalar types (int or float).
    - PT: Type variable for Point2D types.
    - PLT: Type variable for Point2DList types.

Type Aliases:
    - AnyPoint2D: Type alias for a Point2D with either int or float coordinates.
    - IntPoint2D: Type alias for a Point2D with int coordinates.
    - FloatPoint2D: Type alias for a Point2D with float coordinates.
    - AnyPoint2DList: Type alias for a list of Point2D objects with either int or float coordinates.
    - IntPoint2DList: Type alias for a list of Point2D objects with int coordinates.
    - FloatPoint2DList: Type alias for a list of Point2D objects with float coordinates.
    - AnyPoint2DNestedList: Type alias for a nested list of Point2DList objects with either int or float coordinates.
    - IntPoint2DNestedList: Type alias for a nested list of Point2DList objects with int coordinates.
    - FloatPoint2DNestedList: Type alias for a nested list of Point2DList objects with float coordinates.
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from aib.cnt.b_list import BaseList
from aib.cv.geom.b_geom import BaseGeom

if TYPE_CHECKING:
    AnyPoint2D: TypeAlias = Union["Point2D[int]", "Point2D[float]"]
    IntPoint2D: TypeAlias = "Point2D[int]"
    FloatPoint2D: TypeAlias = "Point2D[float]"
    AnyPoint2DList: TypeAlias = Union["Point2DList[IntPoint2D]", "Point2DList[FloatPoint2D]"]
    IntPoint2DList: TypeAlias = "Point2DList[IntPoint2D]"
    FloatPoint2DList: TypeAlias = "Point2DList[FloatPoint2D]"
    AnyPoint2DNestedList: TypeAlias = Union["Point2DNestedList[IntPoint2DList]", "Point2DNestedList[FloatPoint2DList]"]
    IntPoint2DNestedList: TypeAlias = "Point2DNestedList[IntPoint2DList]"
    FloatPoint2DNestedList: TypeAlias = "Point2DNestedList[FloatPoint2DList]"
else:
    AnyPoint2D = Union["Point2D[int]", "Point2D[float]"]
    IntPoint2D = "Point2D[int]"
    FloatPoint2D = "Point2D[float]"
    AnyPoint2DList = Union["Point2DList[IntPoint2D]", "Point2DList[FloatPoint2D]"]
    IntPoint2DList = "Point2DList[IntPoint2D]"
    FloatPoint2DList = "Point2DList[FloatPoint2D]"
    AnyPoint2DNestedList = Union["Point2DNestedList[IntPoint2DList]", "Point2DNestedList[FloatPoint2DList]"]
    IntPoint2DNestedList = "Point2DNestedList[IntPoint2DList]"
    FloatPoint2DNestedList = "Point2DNestedList[FloatPoint2DList]"

T = TypeVar("T", bound=Union[int, float], default=float)
PT = TypeVar("PT", bound=AnyPoint2D, default=FloatPoint2D)
PLT = TypeVar("PLT", bound=AnyPoint2DList, default=FloatPoint2DList)


@dataclass(frozen=True)
class Point2D(BaseGeom, Generic[T]):
    """Point2D Data Class

    Represents a point in 2D space with x and y coordinates.
    Supports conversion to NumPy arrays and various operations like distance calculation, interpolation, and more.

    Attributes:
        x (int | float): The x coordinate of the point.
        y (int | float): The y coordinate of the point.
    """

    x: T = field(compare=True)
    y: T = field(compare=True)

    def to_numpy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the point to a NumPy array.

        Returns:
            npt.NDArray[np.floating | np.integer]: A NumPy array with the x and y coordinates.
        """
        if isinstance(self.x, (int, np.integer)) or isinstance(self.y, (int, np.integer)):
            return np.asarray([self.x, self.y]).astype(np.int32)
        return np.asarray([self.x, self.y]).astype(np.float32)

    def to_int(self) -> IntPoint2D:
        """Convert the point to a Point2D of integers.

        Returns:
            IntPoint2D: A Point2D containing the x and y coordinates as integers.
        """
        return Point2D[int](int(self.x), int(self.y))

    def to_float(self) -> FloatPoint2D:
        """Convert the point to a Point2D of floats.

        Returns:
            FloatPoint2D: A Point2D containing the x and y coordinates as floats.
        """
        return Point2D[float](float(self.x), float(self.y))

    def __eq__(self, other: object) -> bool:
        """Check equality with another Point2D.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the other object is a Point2D with the same coordinates, False otherwise."""
        if not isinstance(other, Point2D):
            return False
        return self.x == other.x and self.y == other.y

    def __iter__(self):
        """Iterate over the x and y coordinates.

        Yields:
            _T: The x and y coordinates of the point."""
        yield self.x
        yield self.y

    def __getitem__(self, a_index: int):
        """Get the x or y coordinate by index.

        Args:
            a_index (int): The index to access (0 for x, 1 for y).

        Returns:
            _T: The x or y coordinate.

        Raises:
            IndexError: If the index is not 0 or 1."""
        if a_index == 0:
            return self.x
        if a_index == 1:
            return self.y
        raise IndexError(f"Invalid index {a_index}: A `Point2D` object only supports indices 0 (x) and 1 (y).")

    def __hash__(self):
        """Return a hash of the Point2D.

        This allows Point2D to be used as a key in dictionaries or sets.
        The hash is based on the x and y coordinates.

        Returns:
            int: The hash value of the Point2D object.
        """
        return hash((self.x, self.y))

    def velocity(self, a_point: AnyPoint2D) -> Tuple[float, float]:
        """Calculate the velocity vector from this point to another Point2D.

        This method computes the normalized velocity vector, which indicates the argsion and rate of movement
        from the current point to a specified target point. The velocity vector is calculated as the difference
        in coordinates, normalized by the distance.

        Args:
            a_point (AnyPoint2D): The other point to calculate the velocity vector to.

        Returns:
            Tuple[float, float]: The velocity vector as (velocity_x, velocity_y)."""
        dy = a_point.y - self.y
        dx = a_point.x - self.x
        norm = np.sqrt(dy * dy + dx * dx) + 1e-6
        return dy / norm, dx / norm

    def distance(self, a_point: AnyPoint2D) -> float:
        """Calculate the Euclidean distance to another point.

        Args:
            a_point (AnyPoint2D): The other point to calculate the distance to.

        Returns:
            float: The Euclidean distance to the other point.
        """
        distance = float(np.linalg.norm(self.to_xy() - a_point.to_xy()))
        return distance

    def distance_xy(self, a_point: AnyPoint2D) -> Tuple[float, float]:
        """Calculate the distance in x and y coordinates to another Point2D.

        Args:
            a_point (AnyPoint2D): The other Point2D to calculate the distance to.

        Returns:
            Tuple[float, float]: The distance in x and y coordinates."""
        distance_x = float(abs(self.x - a_point.x))
        distance_y = float(abs(self.y - a_point.y))
        return distance_x, distance_y

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatPoint2D: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntPoint2D: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyPoint2D: ...

    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
    ) -> AnyPoint2D:
        """Create a Point2D from x and y coordinates.

        Args:
            a_coordinates (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing the x and y coordinates.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Point2D: A Point2D instance with the specified coordinates.
        """
        if isinstance(a_coords, np.ndarray):
            coords = a_coords.flatten()
        else:
            coords = a_coords
        if len(coords) < 2:
            raise ValueError("Coordinates must contain at least 2 values (x, y) for a single point")

        if a_use_float is None:
            if all(isinstance(c, (int, np.integer)) for c in coords):
                scalar_type = int
            else:
                scalar_type = float
        else:
            scalar_type = float if a_use_float else int

        p = cls(cast(T, scalar_type(coords[0])), cast(T, scalar_type(coords[1])))
        return cast(AnyPoint2D, p)

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the point to a NumPy array of shape (2,).

        Returns:
            npt.NDArray[np.floating | np.integer]: A NumPy array with the x and y coordinates.
        """
        return self.to_numpy()

    def is_int(self) -> bool:
        """Check if the coordinates are integers.

        Returns:
            bool: True if both x and y are integers, False otherwise.
        """
        return isinstance(self.x, int) and isinstance(self.y, int)

    def is_float(self) -> bool:
        """Check if the coordinates are floats.

        Returns:
            bool: True if both x and y are floats, False otherwise.
        """
        return isinstance(self.x, float) and isinstance(self.y, float)


class Point2DList(BaseList[PT]):
    """Point2DList Data Container Class

    A list-like container for Point2D objects, providing additional methods for handling collections of points.

    Attributes:
        data (List[PT]): The list of Point2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[PT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Point2DList",
    ):
        """Initialize a Point2DList.

        Args:
            a_iterable (Optional[Iterable[PT]]): An optional iterable to initialize the list.
            a_max_size (Optional[int]): An optional maximum size for the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    def to_tuple(self) -> Tuple[float, ...]:
        """Convert the list of Point2D objects to a tuple of coordinates.

        Returns:
            Tuple[float, ...]: A tuple containing the x and y coordinates of all points in the list.
        """
        points = ()
        for pnt in self.data:
            points += pnt.to_tuple()
        return points

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of Point2D objects to a NumPy array of shape (n, 2).

        Returns:
            npt.NDArray[np.floating | np.integer]: A NumPy array with the x and y coordinates of all points in the list.
        """
        if len(self):
            return np.vstack([point.to_xy() for point in self])
        return np.empty(shape=(0, 2)).astype(np.float32)

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatPoint2DList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntPoint2DList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyPoint2DList: ...

    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
    ) -> AnyPoint2DList:
        """Create a Point2DList from a sequence of x and y coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing pairs of x and y coordinates.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Point2DList: A Point2DList instance containing Point2D objects created from the coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        points = cls(cast(Iterable[PT], [Point2D.from_xy(coord, a_use_float=a_use_float) for coord in a_coords]))
        return cast(AnyPoint2DList, points)

    @property
    def center(self) -> AnyPoint2D:
        """Calculate the center point of the Point2DList.

        Returns:
            Point2D[int | float]: The center point of the Point2DList.
        """
        pnt = np.mean(self.to_xy().astype(np.float32), axis=0)
        return Point2D[int | float](x=pnt[0], y=pnt[1])


class Point2DNestedList(BaseList[PLT]):
    """Point2DNestedList Data Container Class

    A list-like container for lists of Point2D objects, allowing for nested structures of points.

    Attributes:
        data (List[PLT]): The list of Point2DList objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[PLT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Point2DNestedList",
    ):
        """Initialize a Point2DNestedList.

        Args:
            a_iterable (Optional[Iterable[PLT]]): An optional iterable to initialize the nested list.
            a_max_size (Optional[int]): An optional maximum size for the nested list.
            a_name (str): The name of the nested list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatPoint2DNestedList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntPoint2DNestedList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyPoint2DNestedList: ...

    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
    ) -> AnyPoint2DNestedList:
        """Create a Point2DNestedList from a sequence of x and y coordinates.

        Args:
            a_coordinates (Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing lists of x and y coordinates.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Point2DNestedList:
                A Point2DNestedList instance containing Point2DList objects created from the coordinates.
        """
        points = cls()
        points += cast(
            Iterable[PLT], [Point2DList.from_xy(a_coords=coord, a_use_float=a_use_float) for coord in a_coordinates]
        )
        return cast(AnyPoint2DNestedList, points)

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the nested list of Point2DList objects to a NumPy array of shape (n, 2).

        Returns:
            npt.NDArray[np.floating | np.integer]:
                A NumPy array with the x and y coordinates of all points in the nested list.
        """
        return np.vstack([point.to_xy() for point in self])


if not TYPE_CHECKING:
    IntPoint2D = Point2D[int]
    FloatPoint2D = Point2D[float]
    IntPoint2DList = Point2DList[IntPoint2D]
    FloatPoint2DList = Point2DList[FloatPoint2D]
    IntPoint2DNestedList = Point2DNestedList[IntPoint2DList]
    FloatPoint2DNestedList = Point2DNestedList[FloatPoint2DList]
