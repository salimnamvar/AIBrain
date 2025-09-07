"""Computer Vision - Geometry - KeyPoint2D Utilities

This module provides utilities for handling keypoints in 2D space, extending the Point2D class to include a score
attribute.

Classes:
    - KeyPoint2D: Represents a keypoint in 2D space with x, y coordinates and an optional score.
    - KeyPoint2DList: A list-like container for KeyPoint2D objects, allowing for operations on collections of keypoints.

Type Variables:
    - T: Type variable for scalar types (int or float)
    - KPT: Type variable for KeyPoint2D types

Type Aliases:
    - AnyKeyPoint2D: Type alias for any KeyPoint2D type
    - IntKeyPoint2D: Type alias for KeyPoint2D with int coordinates
    - FloatKeyPoint2D: Type alias for KeyPoint2D with float coordinates
    - AnyKeyPoint2DList: Type alias for a list of any KeyPoint2D type
    - IntKeyPoint2DList: Type alias for a list of KeyPoint2D with int coordinates
    - FloatKeyPoint2DList: Type alias for a list of KeyPoint2D with float coordinates
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence, TypeAlias, TypeVar, Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.utils.cv.geom.point.point2d import Point2D, Point2DList, Point2DNestedList

if TYPE_CHECKING:
    AnyKeyPoint2D: TypeAlias = Union["KeyPoint2D[int]", "KeyPoint2D[float]"]
    IntKeyPoint2D: TypeAlias = "KeyPoint2D[int]"
    FloatKeyPoint2D: TypeAlias = "KeyPoint2D[float]"
    AnyKeyPoint2DList: TypeAlias = Union["KeyPoint2DList[IntKeyPoint2D]", "KeyPoint2DList[FloatKeyPoint2D]"]
    IntKeyPoint2DList: TypeAlias = "KeyPoint2DList[IntKeyPoint2D]"
    FloatKeyPoint2DList: TypeAlias = "KeyPoint2DList[FloatKeyPoint2D]"
    AnyKeyPoint2DNestedList: TypeAlias = Union[
        "KeyPoint2DNestedList[IntKeyPoint2DList]", "KeyPoint2DNestedList[FloatKeyPoint2DList]"
    ]
    IntKeyPoint2DNestedList: TypeAlias = "KeyPoint2DNestedList[IntKeyPoint2DList]"
    FloatKeyPoint2DNestedList: TypeAlias = "KeyPoint2DNestedList[FloatKeyPoint2DList]"
else:
    AnyKeyPoint2D = Union["KeyPoint2D[int]", "KeyPoint2D[float]"]
    IntKeyPoint2D = "KeyPoint2D[int]"
    FloatKeyPoint2D = "KeyPoint2D[float]"
    AnyKeyPoint2DList = Union["KeyPoint2DList[IntKeyPoint2D]", "KeyPoint2DList[FloatKeyPoint2D]"]
    IntKeyPoint2DList = "KeyPoint2DList[IntKeyPoint2D]"
    FloatKeyPoint2DList = "KeyPoint2DList[FloatKeyPoint2D]"
    AnyKeyPoint2DNestedList = Union[
        "KeyPoint2DNestedList[IntKeyPoint2DList]", "KeyPoint2DNestedList[FloatKeyPoint2DList]"
    ]
    IntKeyPoint2DNestedList = "KeyPoint2DNestedList[IntKeyPoint2DList]"
    FloatKeyPoint2DNestedList = "KeyPoint2DNestedList[FloatKeyPoint2DList]"


T = TypeVar("T", bound=Union[int, float], default=float)
KPT = TypeVar("KPT", bound=AnyKeyPoint2D, default=FloatKeyPoint2D)
KPLT = TypeVar("KPLT", bound=AnyKeyPoint2DList, default=FloatKeyPoint2DList)


@dataclass(frozen=True)
class KeyPoint2D(Point2D[T]):
    """KeyPoint2D Data Class

    Represents a keypoint in 2D space with x, y coordinates and an optional score.
    This class extends Point2D to include a score attribute.

    Attributes:
        x (float | int): The x coordinate of the keypoint.
        y (float | int): The y coordinate of the keypoint.
        score (Optional[float]): An optional score associated with the keypoint.
    """

    score: Optional[float] = field(default=None, compare=True)

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatKeyPoint2D: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntKeyPoint2D: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyKeyPoint2D: ...

    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
    ) -> AnyKeyPoint2D:
        """Create a KeyPoint2D from x, y, and optional score coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing x, y, and optionally score coordinates.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            KeyPoint2D[_T]: A KeyPoint2D instance with the specified coordinates.

        Raises:
            ValueError: If a_coordinates does not have exactly two (x, y) or three (x, y, score) elements.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.shape[0] == 1:
            coords = a_coords.flatten()
        else:
            coords = a_coords

        if len(coords) == 2:
            x = float(coords[0])
            y = float(coords[1])
            score = None
        elif len(coords) >= 3:
            x = float(coords[0])
            y = float(coords[1])
            score = float(coords[2])
        else:
            raise ValueError(
                f"a_coordinates must have exactly two (x, y) or three (x, y, score) elements, " f"got {a_coords}"
            )
        if a_use_float is None:
            if all([isinstance(x, (int, np.integer)), isinstance(y, (int, np.integer))]):
                scalar_type = int
            else:
                scalar_type = float
        else:
            scalar_type = float if a_use_float else int

        p = cls(cast(T, scalar_type(x)), cast(T, scalar_type(y)), score=score)
        return cast(AnyKeyPoint2D, p)


class KeyPoint2DList(Point2DList[KPT]):
    """KeyPoint2DList Data Container Class

    A list-like container for KeyPoint2D objects, allowing for operations on collections of keypoints.
    Supports initialization from an iterable of KeyPoint2D objects and conversion to NumPy arrays.

    Attributes:
        data (List[KPT]): The list of KeyPoint2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[KPT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "KeyPoint2DList",
    ):
        """Initialize a KeyPoint2DList.

        Args:
            a_iterable (Optional[Iterable[KPT]]): An optional iterable to initialize the list.
            a_max_size (Optional[int]): An optional maximum size for the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatKeyPoint2DList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntKeyPoint2DList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyKeyPoint2DList: ...

    @classmethod
    def from_xy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool],
    ) -> AnyKeyPoint2DList:
        """Create a KeyPoint2DList from a sequence of x, y, and optional score coordinates.

        Args:
            a_coords (Sequence[Sequence[float]] | npt.NDArray[np.float32]):
                A sequence or NumPy array containing x, y coordinates, or a 2D array of shape (N, 2) or (N, 3).
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            KeyPoint2DList: A KeyPoint2DList instance containing KeyPoint2D objects created from the coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        points = cls(
            cast(Iterable[KPT], [KeyPoint2D.from_xy(a_coords=coord, a_use_float=a_use_float) for coord in a_coords])
        )
        return cast(AnyKeyPoint2DList, points)

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the KeyPoint2DList to a NumPy array of shape (N, 3) where N is the number of points.

        Returns:
            npt.NDArray[np.floating | np.integer]: A NumPy array of shape (N, 3) containing the x, y coordinates and score.
        """
        if len(self):
            return np.vstack([point.to_xy() for point in self])
        return np.empty(shape=(0, 3)).astype(np.float32)


class KeyPoint2DNestedList(Point2DNestedList[KPLT]):
    """KeyPoint2DNestedList Data Container Class

    A container for a nested list of KeyPoint2DList objects.

    Attributes:
        data (List[KeyPoint2DList]): A list of KeyPoint2DList objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[KPLT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "KeyPoint2DNestedList",
    ):
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
    ) -> FloatKeyPoint2DNestedList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
    ) -> IntKeyPoint2DNestedList: ...

    @overload
    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
    ) -> AnyKeyPoint2DNestedList: ...

    @classmethod
    def from_xy(
        cls,
        a_coordinates: Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
    ) -> AnyKeyPoint2DNestedList:
        """Create a KeyPoint2DNestedList from a sequence of x and y coordinates and scores.

        Args:
            a_coordinates (Sequence[Sequence[Sequence[float | int]]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing lists of x and y coordinates and scores.
            a_use_float (Optional[bool]):
                If True, use KeyPoint2D[float]; otherwise, use KeyPoint2D[int]. If None, inferred from coords.

        Returns:
            KeyPoint2DNestedList:
                A KeyPoint2DNestedList instance containing KeyPoint2DList objects created from the coordinates.
        """
        points = cls()
        points += cast(
            Iterable[KPLT], [KeyPoint2DList.from_xy(a_coords=coord, a_use_float=a_use_float) for coord in a_coordinates]
        )
        return cast(AnyKeyPoint2DNestedList, points)

    def to_xy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the nested list of KeyPoint2DList objects to a NumPy array of shape (n, 3).

        Returns:
            npt.NDArray[np.floating | np.integer]:
                A NumPy array with the x and y coordinates and scores of all points in the nested list.
        """
        return np.vstack([point.to_xy() for point in self])


if not TYPE_CHECKING:
    IntKeyPoint2D = KeyPoint2D[int]
    FloatKeyPoint2D = KeyPoint2D[float]
    IntKeyPoint2DList = KeyPoint2DList[IntKeyPoint2D]
    FloatKeyPoint2DList = KeyPoint2DList[FloatKeyPoint2D]
    IntKeyPoint2DNestedList = KeyPoint2DNestedList[IntKeyPoint2DList]
    FloatKeyPoint2DNestedList = KeyPoint2DNestedList[FloatKeyPoint2DList]
