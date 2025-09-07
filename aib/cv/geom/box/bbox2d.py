"""Computer Vision - Geometry - BBox2D Utilities

This module provides utilities for working with 2D bounding boxes (BBox2D) in computer vision applications.
It includes classes for representing bounding boxes, lists of bounding boxes, and nested lists of bounding boxes.
The BBox2D class supports various coordinate formats (xyxy, xywh, cxyar, cxywh) and provides methods for conversion,
validation, and manipulation of bounding boxes.

Classes:
    - BBox2D: Represents a 2D bounding box with two corner points, a score, and a label.
    - BBox2DList: A list-like container for BBox2D objects.
    - BBox2DNestedList: A nested list-like container for BBox2DList objects.

Type Variables:
    - T: Type variable for numeric types (int or float).
    - PT: Type variable for Point2D types (int or float).
    - BBT: Type variable for BBox2D types (int or float).
    - BBLT: Type variable for BBox2DList types (int or float).

Type Aliases:
    - AnyBBox2D: Type alias for BBox2D with any Point2D type.
    - IntBBox2D: Type alias for BBox2D with Point2D[int].
    - FloatBBox2D: Type alias for BBox2D with Point2D[float].
    - AnyBBox2DList: Type alias for Union of BBox2DList with BBox2D[Point2D[int]] or BBox2D[Point2D[float]].
    - IntBBox2DList: Type alias for BBox2DList with BBox2D[Point2D[int]].
    - FloatBBox2DList: Type alias for BBox2DList with BBox2D[Point2D[float]].
    - AnyBBox2DNestedList:
        Type alias for Union of BBox2DNestedList with BBox2DList[BBox2D[Point2D[int]]] or
        BBox2DList[BBox2D[Point2D[float]]].
    - IntBBox2DNestedList: Type alias for BBox2DNestedList with BBox2DList[BBox2D[Point2D[int]]].
    - FloatBBox2DNestedList: Type alias for BBox2DNestedList with BBox2DList[BBox2D[Point2D[float]]].
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from aib.cv.geom.box.box2d import Box2D, Box2DList, Box2DNestedList
from aib.cv.geom.box.utils.coord_formats import cxyar_to_xyxy, cxywh_to_xyxy, xywh_to_xyxy
from aib.cv.geom.point.point2d import AnyPoint2D, FloatPoint2D, IntPoint2D, Point2D

if TYPE_CHECKING:
    AnyBBox2D: TypeAlias = Union["BBox2D[IntPoint2D]", "BBox2D[FloatPoint2D]"]
    IntBBox2D: TypeAlias = "BBox2D[IntPoint2D]"
    FloatBBox2D: TypeAlias = "BBox2D[FloatPoint2D]"
    AnyBBox2DList: TypeAlias = Union["BBox2DList[IntBBox2D]", "BBox2DList[FloatBBox2D]"]
    IntBBox2DList: TypeAlias = "BBox2DList[IntBBox2D]"
    FloatBBox2DList: TypeAlias = "BBox2DList[FloatBBox2D]"
    AnyBBox2DNestedList: TypeAlias = Union["BBox2DNestedList[IntBBox2DList]", "BBox2DNestedList[FloatBBox2DList]"]
    IntBBox2DNestedList: TypeAlias = "BBox2DNestedList[IntBBox2DList]"
    FloatBBox2DNestedList: TypeAlias = "BBox2DNestedList[FloatBBox2DList]"
else:
    AnyBBox2D = Union["BBox2D[IntPoint2D]", "BBox2D[FloatPoint2D]"]
    IntBBox2D = "BBox2D[IntPoint2D]"
    FloatBBox2D = "BBox2D[FloatPoint2D]"
    AnyBBox2DList = Union["BBox2DList[IntBBox2D]", "BBox2DList[FloatBBox2D]"]
    IntBBox2DList = "BBox2DList[IntBBox2D]"
    FloatBBox2DList = "BBox2DList[FloatBBox2D]"
    AnyBBox2DNestedList = Union["BBox2DNestedList[IntBBox2DList]", "BBox2DNestedList[FloatBBox2DList]"]
    IntBBox2DNestedList = "BBox2DNestedList[IntBBox2DList]"
    FloatBBox2DNestedList = "BBox2DNestedList[FloatBBox2DList]"

T = TypeVar("T", bound=Union[int, float], default=float)
PT = TypeVar("PT", bound=AnyPoint2D, default=FloatPoint2D)
BBT = TypeVar("BBT", bound=AnyBBox2D, default=FloatBBox2D)
BBLT = TypeVar("BBLT", bound=AnyBBox2DList, default=FloatBBox2DList)


@dataclass(frozen=True)
class BBox2D(Box2D[PT]):
    """BBox2D Data Class

    Represents a 2D bounding box with two corner points, a score, and a label.

    Attributes:
        p1 (PT): The first corner point of the bounding box.
        p2 (PT): The second corner point of the bounding box.
        score (float): The confidence score of the bounding box.
        label (int): The label associated with the bounding box.
    """

    score: float = field(compare=True)
    label: int = field(compare=True)

    def is_score_valid(self, a_score_thre: float) -> bool:
        """Check if the score is above a given threshold.

        Args:
            a_score_thre (float): The score threshold to compare against.

        Returns:
            bool: True if the score is greater than or equal to the threshold, False otherwise.

        Raises:
            TypeError: If a_score_thre is not a float.
        """
        if not isinstance(a_score_thre, (float, np.floating)):
            raise TypeError(f"a_score_thre must be a float, but got {type(a_score_thre)}.")
        return self.score >= a_score_thre

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> FloatBBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> IntBBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D:
        """Create a BBox2D instance from coordinates in various formats.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                The coordinates in one of the supported formats.
            a_coord_format (Literal['xyxy', 'xywh', 'cxywh', 'cxyar']):
                The format of the coordinates. Defaults to "xyxy".
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float): The confidence score for the bounding box. Defaults to 0.0.
            a_label (int): The label for the bounding box. Defaults to 0.

        Returns:
            BBox2D: A BBox2D instance created from the provided coordinates.
        """
        if a_coord_format == "xyxy":
            return cls.from_xyxy(a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, **kwargs)
        if a_coord_format == "xywh":
            return cls.from_xywh(a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, **kwargs)
        if a_coord_format == "cxywh":
            return cls.from_cxywh(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, **kwargs
            )
        if a_coord_format == "cxyar":
            return cls.from_cxyar(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, **kwargs
            )
        raise ValueError(f"Unknown coord_format: {a_coord_format}")

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> FloatBBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> IntBBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D:
        """Create a BBox2D from xyxy coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.

        Returns:
            BBox2D: A BBox2D object with the specified coordinates.
        """
        p1 = Point2D.from_xy(a_coords[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(a_coords[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, **kwargs)
        return cast(AnyBBox2D, box)

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to xyxy coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in xyxy format.
        """
        return np.concatenate((self.p1.to_xy(), self.p2.to_xy(), np.array([self.score, self.label], dtype=np.float32)))

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> FloatBBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> IntBBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D:
        """Create a BBox2D from xywh coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]):
                The coordinates in xywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.

        Returns:
            BBox2D: A BBox2D object with the specified coordinates.
        """
        xyxy = xywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, **kwargs)
        return cast(AnyBBox2D, box)

    def to_xywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to xywh coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in xywh format.
        """
        return np.concatenate((self.p1.to_xy(), self.size.to_numpy(), np.array([self.score, self.label])))

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> FloatBBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> IntBBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D:
        """Create a BBox2D from cxyar coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]): The coordinates in cxyar format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.

        Returns:
            BBox2D: A BBox2D object with the specified coordinates.
        """
        xyxy = cxyar_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, **kwargs)
        return cast(AnyBBox2D, box)

    def to_cxyar(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to cxyar coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in cxyar format.
        """
        return np.concatenate(
            (
                self.center.to_numpy(),
                np.array([self.area, self.aspect_ratio, self.score, self.label]),
            )
        )

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> FloatBBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> IntBBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        **kwargs: Any,
    ) -> AnyBBox2D:
        """Create a BBox2D from cxywh coordinates.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.floating | np.integer]): The coordinates in cxywh format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.

        Returns:
            BBox2D: A BBox2D object with the specified coordinates.
        """
        xyxy = cxywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, **kwargs)
        return cast(AnyBBox2D, box)

    def to_cxywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the bounding box to cxywh coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding box in cxywh format.
        """
        return np.concatenate((self.center.to_xy(), self.size.to_numpy(), np.array([self.score, self.label])))

    def __iter__(self) -> Iterable[Any]:
        """Iterate over the bounding box coordinates, score, and label.

        Yields:
            Iterable[Any]: The coordinates (x1, y1, x2, y2), score, and label.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y
        yield self.score
        yield self.label

    def __getitem__(self, a_index: int) -> Any:
        """Get the value at the specified index.

        Args:
            a_index (int): The index to access.

        Returns:
            Any: The value at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if a_index == 0:
            return self.p1.x
        if a_index == 1:
            return self.p1.y
        if a_index == 2:
            return self.p2.x
        if a_index == 3:
            return self.p2.y
        if a_index == 4:
            return self.score
        if a_index == 5:
            return self.label
        raise IndexError(
            f"Invalid index {a_index}: A `BBox2D` object only supports indices "
            f"0 (x1), 1 (y1), 2 (x2), 3 (y2), 4(score), 5(label)."
        )

    def to_box2d(self) -> Box2D[PT]:
        """Convert the BBox2D to a Box2D object.

        Returns:
            Box2D[PT]: A Box2D object with the same coordinates.
        """
        return Box2D(p1=self.p1, p2=self.p2)


class BBox2DList(Box2DList[BBT]):
    """BBox2DList Data Container Class

    A list-like container for BBox2D objects, allowing for operations on a collection of bounding boxes.

    Attributes:
        data (list[BBT]): The list of BBox2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[BBT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "BBox2DList",
    ):
        """Initialize a BBox2DList.

        Args:
            a_iterable (Optional[Iterable[BBT]]): An iterable of BBox2D objects.
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
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> FloatBBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> IntBBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList:
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
            return cls.from_xyxy(
                a_coords=a_coords, a_use_float=a_use_float, a_scores=a_scores, a_labels=a_labels, **kwargs
            )
        if a_coord_format == "xywh":
            return cls.from_xywh(
                a_coords=a_coords, a_use_float=a_use_float, a_scores=a_scores, a_labels=a_labels, **kwargs
            )
        if a_coord_format == "cxywh":
            return cls.from_cxywh(
                a_coords=a_coords, a_use_float=a_use_float, a_scores=a_scores, a_labels=a_labels, **kwargs
            )
        if a_coord_format == "cxyar":
            return cls.from_cxyar(
                a_coords=a_coords, a_use_float=a_use_float, a_scores=a_scores, a_labels=a_labels, **kwargs
            )
        raise ValueError(f"Unknown coord_format: {a_coord_format}")

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> FloatBBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> IntBBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList:
        """Create a BBox2DList from xyxy coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in xyxy format.
            a_use_float: If True, use Point2D[float]; otherwise, use Point2D[int].
                        If None, inferred from coords.
            a_scores (Optional[Sequence[float] | npt.NDArray[np.floating]]):
                Optional scores for each bounding box. If not provided, defaults to 0.0 or extracted from the
                coordinates.
            a_labels (Optional[Sequence[int] | npt.NDArray[np.integer]]):
                Optional labels for each bounding box. If not provided, defaults to 0 or extracted from the coordinates.

        Returns:
            BBox2DList: A BBox2DList object with the specified coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        bboxes: List[AnyBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            bbox = BBox2D.from_xyxy(a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, **kwargs)
            bboxes.append(bbox)
        return cast(AnyBBox2DList, cls(cast(Iterable[BBT], bboxes)))

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of BBox2D to xyxy coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in xyxy format.
        """
        if len(self):
            return np.vstack([box.to_xyxy() for box in self])
        return np.empty(shape=(0, 6), dtype=np.float32)

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> FloatBBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> IntBBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList:
        """Create a BBox2DList from xywh coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in xywh format.
            a_use_float: If True, use Point2D[float]; otherwise, use Point2D[int].
                        If None, inferred from coords.
            a_scores (Optional[Sequence[float] | npt.NDArray[np.floating]]):
                Optional scores for each bounding box. If not provided, defaults to 0.0 or extracted from the
                coordinates.
            a_labels (Optional[Sequence[int] | npt.NDArray[np.integer]]):
                Optional labels for each bounding box. If not provided, defaults to 0 or extracted from the coordinates.

        Returns:
            BBox2DList: A BBox2DList object with the specified coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        bboxes: List[AnyBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            bbox = BBox2D.from_xywh(a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, **kwargs)
            bboxes.append(bbox)
        return cast(AnyBBox2DList, cls(cast(Iterable[BBT], bboxes)))

    def to_xywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of BBox2D to xywh coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in xywh format.
        """
        if len(self):
            return np.vstack([box.to_xywh() for box in self])
        return np.empty(shape=(0, 6), dtype=np.float32)

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> FloatBBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> IntBBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList:
        """Create a BBox2DList from cxyar coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in cxyar format.
            a_use_float: If True, use Point2D[float]; otherwise, use Point2D[int].
                        If None, inferred from coords.
            a_scores (Optional[Sequence[float] | npt.NDArray[np.floating]]):
                Optional scores for each bounding box. If not provided, defaults to 0.0 or extracted from the
                coordinates.
            a_labels (Optional[Sequence[int] | npt.NDArray[np.integer]]):
                Optional labels for each bounding box. If not provided, defaults to 0 or extracted from the coordinates.

        Returns:
            BBox2DList: A BBox2DList object with the specified coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        bboxes: List[AnyBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            bbox = BBox2D.from_cxyar(a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, **kwargs)
            bboxes.append(bbox)
        return cast(AnyBBox2DList, cls(cast(Iterable[BBT], bboxes)))

    def to_cxyar(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of BBox2D to cxyar coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in cxyar format.
        """
        if len(self):
            return np.vstack([box.to_cxyar() for box in self])
        return np.empty(shape=(0, 6), dtype=np.float32)

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> FloatBBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> IntBBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        **kwargs: Any,
    ) -> AnyBBox2DList:
        """Create a BBox2DList from cxywh coordinates.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                The coordinates in cxywh format.
            a_use_float: If True, use Point2D[float]; otherwise, use Point2D[int].
                        If None, inferred from coords.
            a_scores (Optional[Sequence[float] | npt.NDArray[np.floating]]):
                Optional scores for each bounding box. If not provided, defaults to 0.0 or extracted from the
                coordinates.
            a_labels (Optional[Sequence[int] | npt.NDArray[np.integer]]):
                Optional labels for each bounding box. If not provided, defaults to 0 or extracted from the coordinates.

        Returns:
            BBox2DList: A BBox2DList object with the specified coordinates.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        bboxes: List[AnyBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            bbox = BBox2D.from_cxywh(a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, **kwargs)
            bboxes.append(bbox)
        return cast(AnyBBox2DList, cls(cast(Iterable[BBT], bboxes)))

    def to_cxywh(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the list of BBox2D to cxywh coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: The bounding boxes in cxywh format.
        """
        if len(self):
            return np.vstack([box.to_cxywh() for box in self])
        return np.empty(shape=(0, 6), dtype=np.float32)


class BBox2DNestedList(Box2DNestedList[BBLT]):
    """BBox2DNestedList Data Container Class

    A nested list-like container for BBox2DList objects, allowing for operations on collections of bounding boxes.

    Attributes:
        data (list[BBox2DList[PT]]): The list of BBox2DList objects."""

    def __init__(
        self,
        a_iterable: Optional[Iterable[BBLT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "BBox2DNestedList",
    ):
        """Initialize a BBox2DNestedList.

        Args:
            a_iterable (Optional[Iterable[BBLT]]): An iterable of BBox2DList objects.
            a_max_size (Optional[int]): The maximum size of the nested list.
            a_name (str): The name of the nested list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    def to_xyxy(self) -> npt.NDArray[np.floating | np.integer]:
        """Convert the nested list of BBox2DList to xyxy coordinates.

        Returns:
            npt.NDArray[np.floating | np.integer]: A 2D array of bounding boxes in xyxy format.
        """
        if len(self):
            return np.vstack([box_list.to_xyxy() for box_list in self])
        return np.empty(shape=(0, 6), dtype=np.float32)

    def to_bbox2dlist(self) -> BBLT:
        """Convert the nested list of BBox2DList to a flat BBox2DList.

        Returns:
            BBLT: A flat BBox2DList containing all bounding boxes from the nested lists.
        """
        boxes: BBox2DList[BBox2D[Point2D[float]] | BBox2D[Point2D[int]]] = BBox2DList()
        for item in self:
            for box in item:
                boxes.append(box)
        return cast(BBLT, boxes)


if not TYPE_CHECKING:
    IntBBox2D = BBox2D[IntPoint2D]
    FloatBBox2D = BBox2D[FloatPoint2D]
    IntBBox2DList = BBox2DList[IntBBox2D]
    FloatBBox2DList = BBox2DList[FloatBBox2D]
    IntBBox2DNestedList = BBox2DNestedList[IntBBox2DList]
    FloatBBox2DNestedList = BBox2DNestedList[FloatBBox2DList]
