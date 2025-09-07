"""Computer Vision - Geometry - Box2D Utilities

This module provides utilities for working with 2D bounding boxes, including creation, manipulation, and conversion
between different coordinate formats.

Classes:
    - Box2D: Represents a 2D bounding box defined by two points (p1 and p2).
    - Box2DList: A list-like container for Box2D objects.
    - Box2DNestedList: A list-like container for lists of Box2D objects.

Type Variables:
    - T: Type variable for numeric types (int or float).
    - PT: Type variable for Point2D types (int or float).
    - BT: Type variable for Box2D types (int or float).
    - BLT: Type variable for Box2DList types (int or float).

Type Aliases:
    - AnyBox2D: Type alias for Box2D with any Point2D type.
    - IntBox2D: Type alias for Box2D with Point2D[int].
    - FloatBox2D: Type alias for Box2D with Point2D[float].
    - AnyBox2DList: Type alias for Box2DList with any Point2D type.
    - IntBox2DList: Type alias for Box2DList with Point2D[int].
    - FloatBox2DList: Type alias for Box2DList with Point2D[float].
    - AnyBox2DNestedList: Type alias for Box2DNestedList with any Point2D type.
    - IntBox2DNestedList: Type alias for Box2DNestedList with Point2D[int].
    - FloatBox2DNestedList: Type alias for Box2DNestedList with Point2D[float].
"""

import math
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Literal,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from pyparsing import Union

from src.utils.cnt.b_list import BaseList
from src.utils.cv.geom.b_geom import BaseGeom
from src.utils.cv.geom.box.utils.coord_formats import cxyar_to_xyxy, cxywh_to_xyxy, xywh_to_xyxy
from src.utils.cv.geom.point.point2d import AnyPoint2D, FloatPoint2D, IntPoint2D, Point2D
from src.utils.cv.geom.size import AnySize, Size

if TYPE_CHECKING:
    AnyBox2D: TypeAlias = Union["Box2D[IntPoint2D]", "Box2D[FloatPoint2D]"]
    IntBox2D: TypeAlias = "Box2D[IntPoint2D]"
    FloatBox2D: TypeAlias = "Box2D[FloatPoint2D]"
    AnyBox2DList: TypeAlias = Union["Box2DList[IntBox2D]", "Box2DList[FloatBox2D]"]
    IntBox2DList: TypeAlias = "Box2DList[IntBox2D]"
    FloatBox2DList: TypeAlias = "Box2DList[FloatBox2D]"
    AnyBox2DNestedList: TypeAlias = Union["Box2DNestedList[IntBox2DList]", "Box2DNestedList[FloatBox2DList]"]
    IntBox2DNestedList: TypeAlias = "Box2DNestedList[IntBox2DList]"
    FloatBox2DNestedList: TypeAlias = "Box2DNestedList[FloatBox2DList]"
else:
    AnyBox2D = Union["Box2D[IntPoint2D]", "Box2D[FloatPoint2D]"]
    IntBox2D = "Box2D[IntPoint2D]"
    FloatBox2D = "Box2D[FloatPoint2D]"
    AnyBox2DList = Union["Box2DList[IntBox2D]", "Box2DList[FloatBox2D]"]
    IntBox2DList = "Box2DList[IntBox2D]"
    FloatBox2DList = "Box2DList[FloatBox2D]"
    AnyBox2DNestedList = Union["Box2DNestedList[IntBox2DList]", "Box2DNestedList[FloatBox2DList]"]
    IntBox2DNestedList = "Box2DNestedList[IntBox2DList]"
    FloatBox2DNestedList = "Box2DNestedList[FloatBox2DList]"

T = TypeVar("T", bound=Union[int, float], default=float)
PT = TypeVar("PT", bound=AnyPoint2D, default=FloatPoint2D)
BT = TypeVar("BT", bound=AnyBox2D, default=FloatBox2D)
BLT = TypeVar("BLT", bound=AnyBox2DList, default=FloatBox2DList)


@dataclass(frozen=True)
class Box2D(BaseGeom, Generic[PT]):
    """Box2D Data Class

    Represents a 2D bounding box defined by two points (p1 and p2).

    Attributes:
        p1 (PT): The top-left point of the bounding box.
        p2 (PT): The bottom-right point of the bounding box.
        width (int): The width of the bounding box.
        height (int): The height of the bounding box.
        area (float): The area of the bounding box.
        aspect_ratio (float): The aspect ratio of the bounding box (width / height).
        size (Size[int | float]): The size of the bounding box.
        center (Point2D[int | float]): The center point of the bounding box.
    """

    p1: PT = field(compare=False)
    p2: PT = field(compare=False)

    @property
    def width(self) -> int | float:
        """Calculate the width of the bounding box.

        Returns:
            int: The width of the bounding box (p2.x - p1.x).
        """
        return self.p2.x - self.p1.x

    @property
    def height(self) -> int | float:
        """Calculate the height of the bounding box.

        Returns:
            int: The height of the bounding box (p2.y - p1.y).
        """
        return self.p2.y - self.p1.y

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box.

        Returns:
            float: The area of the bounding box (width * height).
        """
        return self.size.area

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the bounding box.

        Returns:
            float: The aspect ratio of the bounding box (width / height).
        """
        return self.size.aspect_ratio

    @property
    def size(self) -> AnySize:
        """Get the size of the bounding box.

        Returns:
            AnySize: The size of the bounding box, either as integers or floats.
        """
        if self.is_float():
            return Size[float](width=float(self.width), height=float(self.height))
        return Size[int](width=int(self.width), height=int(self.height))

    @property
    def center(self) -> AnyPoint2D:
        """Calculate the center point of the bounding box.

        Returns:
            AnyPoint: The center point of the bounding box.
        """
        x = (self.p1.x + self.p2.x) / 2.0
        y = (self.p1.y + self.p2.y) / 2.0
        if self.is_float():
            return Point2D[float](x=float(x), y=float(y))
        return Point2D[int](int(x), int(y))

    def is_int(self) -> bool:
        """Check if the bounding box coordinates are integers.

        Returns:
            bool: True if both points are integers, False otherwise.
        """
        return self.p1.is_int() and self.p2.is_int()

    def is_float(self) -> bool:
        """Check if the bounding box coordinates are floats.

        Returns:
            bool: True if both points are floats, False otherwise.
        """
        return self.p1.is_float() and self.p2.is_float()

    def is_coord_valid(self) -> bool:
        """Check if the coordinates of the bounding box are valid.

        Returns:
            bool: True if the coordinates are valid (p2 is greater than p1), False otherwise.
        """
        return self.p2.x > self.p1.x and self.p2.y > self.p1.y

    def is_oob_valid(self, a_size: AnySize) -> bool:
        """Check if the bounding box is within the bounds of a given size.

        Args:
            a_size (AnySize): The size to check against.

        Returns:
            bool: True if the bounding box is within the bounds, False otherwise.
        """
        return not (self.p2.x <= 0 or self.p2.y <= 0 or self.p1.x >= a_size.width or self.p1.y >= a_size.height)

    def is_size_valid(self, a_size: AnySize) -> bool:
        """Check if the bounding box size is valid compared to a given size.

        Args:
            a_size (AnySize): The size to compare against.

        Returns:
            bool: True if the bounding box size is valid, False otherwise.
        """
        return self.size >= a_size

    def to_int(self) -> Self:
        """Convert the bounding box coordinates to integers.

        Returns:
            IntBox: A new IntBox instance with integer coordinates.
        """
        return replace(
            self,
            p1=self.p1.to_int(),
            p2=self.p2.to_int(),
        )

    def to_float(self) -> Self:
        """Convert the bounding box coordinates to floats.

        Returns:
            FloatBox: A new FloatBox instance with float coordinates.
        """
        return replace(
            self,
            p1=self.p1.to_float(),
            p2=self.p2.to_float(),
        )

    def intersection(self, a_box: AnyBox2D) -> int:
        """Calculate the intersection area with another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to intersect with.

        Returns:
            int: The area of the intersection of the two bounding boxes.
        """
        x_min, y_min = np.maximum((self.p1.x, self.p1.y), (a_box.p1.x, a_box.p1.y))
        x_max, y_max = np.minimum((self.p2.x, self.p2.y), (a_box.p2.x, a_box.p2.y))
        if x_min >= x_max or y_min >= y_max:
            return 0
        intersection_width = x_max - x_min
        intersection_height = y_max - y_min
        return int(intersection_width * intersection_height)

    def union(self, a_box: AnyBox2D) -> int:
        """Calculate the union area with another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to union with.

        Returns:
            int: The area of the union of the two bounding boxes.
        """
        return int(self.area + a_box.area - self.intersection(a_box))

    def iou(self, a_box: AnyBox2D) -> float:
        """Calculate the Intersection over Union (IoU) with another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to calculate IoU with.

        Returns:
            float: The IoU value, which is the ratio of the intersection area to the union area.
        """
        return self.intersection(a_box) / self.union(a_box)

    def velocity(self, a_box: AnyBox2D) -> Tuple[float, float]:
        """
        Calculate the velocity vector from the center of the current box to another box.

        Args:
            a_box (AnyBox2D): The other bounding box to calculate the velocity from.

        Returns:
            Tuple[float, float]: A tuple representing the velocity vector (velocity_x, velocity_y).
        """
        velocity_x, velocity_y = self.center.velocity(a_box.center)
        return velocity_x, velocity_y

    def distance_centroid(self, a_box: AnyBox2D) -> float:
        """Calculate the distance between the centroids of two bounding boxes.

        Args:
            a_box (AnyBox2D): The other bounding box to calculate the distance from.

        Returns:
            float: The distance between the centroids of the two bounding boxes.
        """
        distance = self.center.to_float().distance(a_box.center.to_float())
        return distance

    def distance_centroid_xy(self, a_box: AnyBox2D) -> Tuple[float, float]:
        """Calculate the distance in x and y argsions between the centroids of two bounding boxes.

        Args:
            a_box (AnyBox2D): The other bounding box to calculate the distance from.

        Returns:
            Tuple[float, float]: The distance in x and y argsions.
        """
        distance_x, distance_y = self.center.to_float().distance_xy(a_box.center.to_float())
        return distance_x, distance_y

    def distance_min(self, a_box: AnyBox2D) -> float:
        """Calculate the minimum distance between the bounding boxes.

        Args:
            a_box (AnyBox2D): The other bounding box to calculate the distance from.

        Returns:
            float: The minimum distance between the two bounding boxes.
        """
        dx, dy = self.distance_min_xy(a_box=a_box)
        return math.sqrt(dx * dx + dy * dy)

    def distance_min_xy(self, a_box: AnyBox2D) -> Tuple[float, float]:
        """Calculate the minimum distance in x and y argsions between the bounding boxes.

        Args:
            a_box (Box2D[PT]): The other bounding box to calculate the distance from.

        Returns:
            Tuple[float, float]: The minimum distance in x and y argsions.
        """
        dx = max(self.p1.x - a_box.p2.x, a_box.p1.x - self.p2.x, 0)
        dy = max(self.p1.y - a_box.p2.y, a_box.p1.y - self.p2.y, 0)
        return dx, dy

    def clamp(self, a_size: AnySize) -> Self:
        """Clamp the bounding box coordinates to fit within a given size.

        Args:
            a_size (AnySize): The size to clamp the bounding box to.

        Returns:
            Box2D[PT]: A new Box2D instance with clamped coordinates.
        """
        return replace(
            self,
            p1=Point2D[type(self.p1.x)](x=max(self.p1.x, type(self.p1.x)(0)), y=max(self.p1.y, type(self.p1.y)(0))),
            p2=Point2D[type(self.p2.x)](
                x=min(self.p2.x, type(self.p2.x)(a_size.width)),
                y=min(self.p2.y, type(self.p2.y)(a_size.height)),
            ),
        )

    def scale(self, a_factor: float, a_by_area: bool = False) -> Self:
        """Scale the bounding box by a given factor.

        Args:
            a_factor (float): The factor by which to scale the bounding box.
            a_by_area (bool): If True, scale by area; otherwise, scale by width and height.

        Returns:
            Box2D[PT]: A new Box2D instance with scaled coordinates.
        """
        if not isinstance(a_factor, (float, np.floating)):
            raise TypeError(f"a_factor must be a float, but got {type(a_factor)}.")
        if not 0 <= a_factor:
            raise ValueError(f"a_factor must be bigger than 0, but got {a_factor}.")
        if a_by_area:
            a_factor = np.sqrt(a_factor)
        width = self.width * a_factor
        height = self.height * a_factor
        return replace(
            self,
            p1=Point2D[type(self.p1.x)](
                x=type(self.p1.x)(self.center.x - width / 2), y=type(self.p1.y)(self.center.y - height / 2)
            ),
            p2=Point2D[type(self.p2.x)](
                x=type(self.p2.x)(self.center.x + width / 2), y=type(self.p2.y)(self.center.y + height / 2)
            ),
        )

    def shift(self, a_size: AnySize) -> Self:
        """Shift the bounding box coordinates by a given size.

        Args:
            a_size (AnySize): The size to shift the bounding box by.

        Returns:
            Box2D[PT]: A new Box2D instance with shifted coordinates.
        """
        return replace(
            self,
            p1=Point2D[type(self.p1.x)](
                x=type(self.p1.x)(self.p1.x + a_size.width), y=type(self.p1.y)(self.p1.y + a_size.height)
            ),
            p2=Point2D[type(self.p2.x)](
                x=type(self.p2.x)(self.p2.x + a_size.width), y=type(self.p2.y)(self.p2.y + a_size.height)
            ),
        )

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2D: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2D:
        """Create a Box2D instance from coordinates in various formats.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                The coordinates in one of the supported formats.
            a_coord_format (Literal['xyxy', 'xywh', 'cxywh', 'cxyar']):
                The format of the coordinates. Defaults to "xyxy".
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2D[PT]: A Box2D instance created from the provided coordinates.
        """
        if a_coord_format == "xyxy":
            return cls.from_xyxy(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "xywh":
            return cls.from_xywh(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "cxywh":
            return cls.from_cxywh(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "cxyar":
            return cls.from_cxyar(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        raise ValueError(f"Unknown coord_format: {a_coord_format}")

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2D: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2D:
        """Create a Box2D instance from xyxy coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]): The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2D[PT]: A Box2D instance created from the provided coordinates.
        """
        p1 = Point2D.from_xy(a_coords[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(a_coords[2:], a_use_float=a_use_float)
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), **kwargs)
        return cast(AnyBox2D, box)

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to xyxy coordinates.

        The coordinates are in the format (x1, y1, x2, y2).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in xyxy format.
        """
        return np.concatenate((self.p1.to_xy(), self.p2.to_xy()))

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2D: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2D:
        """Create a Box2D instance from xywh coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]): The coordinates in xywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2D[PT]: A Box2D instance created from the provided coordinates.
        """
        xyxy = xywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), **kwargs)
        return cast(AnyBox2D, box)

    def to_xywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to xywh coordinates.

        The coordinates are in the format (x1, y1, width, height).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in xywh format.
        """
        return np.concatenate((self.p1.to_xy(), self.size.to_numpy()))

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2D: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2D:
        """Create a Box2D instance from cxyar coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in cxyar format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2D[PT]: A Box2D instance created from the provided coordinates.
        """
        xyxy = cxyar_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), **kwargs)
        return cast(AnyBox2D, box)

    def to_cxyar(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to cxyar coordinates.

        The coordinates are in the format (center_x, center_y, aspect_ratio, area).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in cxyar format.
        """
        return np.concatenate((self.center.to_xy(), [self.area, self.aspect_ratio]))

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2D: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2D:
        """Create a Box2D instance from cxywh coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in cxywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2D[PT]: A Box2D instance created from the provided coordinates.
        """
        xyxy = cxywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), **kwargs)
        return cast(AnyBox2D, box)

    def to_cxywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to cxywh coordinates.

        The coordinates are in the format (center_x, center_y, width, height).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in cxywh format.
        """
        return np.concatenate((self.center.to_xy(), self.size.to_numpy()))

    def __iter__(self) -> Iterable[Any]:
        """Iterate over the coordinates of the bounding box.

        Yields:
            The x and y coordinates of the top-left and bottom-right points of the bounding box.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y

    def __getitem__(self, a_index: int) -> Any:
        """Get the x or y coordinate of the bounding box based on the index.

        Args:
            a_index (int): The index of the coordinate to retrieve (0 for x1, 1 for y1, 2 for x2, 3 for y2).

        Returns:
            Any: The x or y coordinate of the bounding box.

        Raises:
            IndexError: If the index is not 0, 1, 2, or 3.
        """
        if a_index == 0:
            return self.p1.x
        if a_index == 1:
            return self.p1.y
        if a_index == 2:
            return self.p2.x
        if a_index == 3:
            return self.p2.y
        raise IndexError(
            f"Invalid index {a_index}: A `Box2D` object only supports indices 0 (x1), 1 (y1), 2 (x2), 3 (y2)."
        )

    def __add__(self, a_box: AnyBox2D) -> int:
        """Calculate the union area with another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to union with.

        Returns:
            int: The area of the union of the two bounding boxes.
        """
        return self.union(a_box)

    def __sub__(self, a_box: AnyBox2D) -> int:
        """Calculate the difference in area between the bounding box and another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to subtract.

        Returns:
            int: The area of the bounding box minus the intersection area with the other bounding box.
        """
        return int(self.area - self.intersection(a_box))

    def __mul__(self, a_box: AnyBox2D) -> int:
        """Calculate the intersection area with another bounding box.

        Args:
            a_box (AnyBox2D): The other bounding box to intersect with.

        Returns:
            int: The area of the intersection of the two bounding boxes.
        """
        return self.intersection(a_box)

    def __truediv__(self, a_box: AnyBox2D) -> float:
        """Calculate the ratio of the areas of two bounding boxes.

        Args:
            a_box (AnyBox2D): The other bounding box to compare with.

        Returns:
            float: The ratio of the area of the current bounding box to the area of the other bounding box.
        """
        return self.area / a_box.area


class Box2DList(BaseList[BT]):
    """Box2DList Data Container Class

    A list-like container for Box2D objects, providing methods to create and manipulate lists of bounding boxes.

    Attributes:
        data (Iterable[BT]): An iterable of Box2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[BT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Box2DList",
    ):
        """Initialize a Box2DList instance.

        Args:
            a_iterable (Optional[Iterable[BT]]): An iterable of Box2D objects to initialize the list with.
            a_max_size (Optional[int]): The maximum size of the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2DList: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2DList:
        """Create a Box2DList instance from coordinates in various formats.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in one of the supported formats.
            a_coord_format (Literal['xyxy', 'xywh', 'cxywh', 'cxyar']):
                The format of the coordinates. Defaults to "xyxy".
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2DList: A Box2DList instance created from the provided coordinates.
        """
        if a_coord_format == "xyxy":
            return cls.from_xyxy(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "xywh":
            return cls.from_xywh(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "cxywh":
            return cls.from_cxywh(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        if a_coord_format == "cxyar":
            return cls.from_cxyar(a_coords=a_coords, a_use_float=a_use_float, **kwargs)
        raise ValueError(f"Unknown coord_format: {a_coord_format}")

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2DList: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2DList:
        """Create a Box2DList from xyxy coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2DList: A Box2DList instance created from the provided coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        boxes = cls(
            cast(Iterable[BT], [Box2D.from_xyxy(coord, a_use_float=a_use_float, **kwargs) for coord in a_coords])
        )
        return cast(AnyBox2DList, boxes)

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of bounding boxes to xyxy coordinates.

        The coordinates are in the format (x1, y1, x2, y2).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in xyxy format.
        """
        if len(self):
            return np.vstack([box.to_xyxy() for box in self])
        return np.empty(shape=(0, 4)).astype(np.float32)

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2DList: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2DList:
        """Create a Box2DList from xywh coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.float32]): The coordinates in xywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2DList: A Box2DList instance created from the provided coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        bboxes = cls(
            cast(Iterable[BT], [Box2D.from_xywh(coord, a_use_float=a_use_float, **kwargs) for coord in a_coords])
        )
        return cast(AnyBox2DList, bboxes)

    def to_xywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of bounding boxes to xywh coordinates.

        The coordinates are in the format (x1, y1, width, height).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in xywh format.
        """
        if len(self):
            return np.vstack([box.to_xywh() for box in self])
        return np.empty(shape=(0, 4)).astype(np.float32)

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2DList: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2DList:
        """Create a Box2DList from cxyar coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.float32]): The coordinates in cxyar format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2DList: A Box2DList instance created from the provided coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        bboxes = cls(
            cast(Iterable[BT], [Box2D.from_cxyar(coord, a_use_float=a_use_float, **kwargs) for coord in a_coords])
        )
        return cast(AnyBox2DList, bboxes)

    def to_cxyar(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of bounding boxes to cxyar coordinates.

        The coordinates are in the format (center_x, center_y, aspect_ratio, area).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in cxyar format.
        """
        if len(self):
            return np.vstack([box.to_cxyar() for box in self])
        return np.empty(shape=(0, 4)).astype(np.float32)

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        **kwargs: Any,
    ) -> FloatBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        **kwargs: Any,
    ) -> IntBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        **kwargs: Any,
    ) -> AnyBox2DList: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        **kwargs: Any,
    ) -> AnyBox2DList:
        """Create a Box2DList from cxywh coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.float32]): The coordinates in cxywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.

        Returns:
            Box2DList: A Box2DList instance created from the provided coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        bboxes = cls(
            cast(Iterable[BT], [Box2D.from_cxywh(coord, a_use_float=a_use_float, **kwargs) for coord in a_coords])
        )
        return cast(AnyBox2DList, bboxes)

    def to_cxywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of bounding boxes to cxywh coordinates.

        The coordinates are in the format (center_x, center_y, width, height).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in cxywh format.
        """
        if len(self):
            return np.vstack([box.to_cxywh() for box in self])
        return np.empty(shape=(0, 4)).astype(np.float32)

    def clamp(self, a_size: AnySize) -> Self:
        """Clamp the bounding boxes to fit within a given size.

        Args:
            a_size (AnySize): The size to clamp the bounding boxes to.

        Returns:
            Box2DList: A new Box2DList with clamped bounding boxes.
        """
        return self.__class__(box.clamp(a_size=a_size) for box in self.data)

    def scale(self, a_factor: float) -> Self:
        """Scale the bounding boxes by a given factor.

        Args:
            a_factor (float): The factor by which to scale the bounding boxes.

        Returns:
            Box2DList: A new Box2DList with scaled bounding boxes.
        """
        return self.__class__(box.scale(a_factor=a_factor) for box in self.data)

    def shift(self, a_size: AnySize) -> Self:
        """Shift the bounding boxes by a given size.

        Args:
            a_size (AnySize): The size to shift the bounding boxes by.

        Returns:
            Box2DList: A new Box2DList with shifted bounding boxes.
        """
        return self.__class__(box.shift(a_size=a_size) for box in self.data)


class Box2DNestedList(BaseList[BLT]):
    """Box2DNestedList Data Container Class

    A list-like container for lists of Box2D objects, providing methods to create and manipulate nested lists of
    bounding boxes.

    Attributes:
        data (Iterable[BLT]): An iterable of Box2DList objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[BLT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Box2DNestedList",
    ):
        """Initialize a Box2DNestedList instance.

        Args:
            a_iterable (Optional[Iterable[BLT]]): An iterable of Box2DList objects to initialize the nested list with.
            a_max_size (Optional[int]): The maximum size of the nested list.
            a_name (str): The name of the nested list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the nested list of bounding boxes to xyxy coordinates.

        The coordinates are in the format (x1, y1, x2, y2).

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in xyxy format.
        """
        if len(self):
            return np.vstack([box_list.to_xyxy() for box_list in self])
        return np.empty(shape=(0, 5)).astype(np.float32)


if not TYPE_CHECKING:
    IntBox2D = Box2D[IntPoint2D]
    FloatBox2D = Box2D[FloatPoint2D]
    IntBox2DList = Box2DList[IntBox2D]
    FloatBox2DList = Box2DList[FloatBox2D]
    IntBox2DNestedList = Box2DNestedList[IntBox2DList]
    FloatBox2DNestedList = Box2DNestedList[FloatBox2DList]
