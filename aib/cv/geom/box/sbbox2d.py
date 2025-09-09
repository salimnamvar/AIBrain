"""Computer Vision - Geometry - SegBBox2D Utilities

This module provides utilities for handling segmentation bounding boxes in computer vision tasks.
It defines classes for 2D segmentation bounding boxes, including methods for clamping, scaling, and converting
coordinates.

Classes:
    - SegBBox2D:
        Represents a 2D bounding box with segmentation mask, providing methods for clamping, scaling, and creating from
        various coordinate formats.
    - SegBBox2DList:
        A list-like container for SegBBox2D objects, allowing operations on collections of segmentation bounding boxes.
    - SegBBox2DNestedList:
        A nested list-like container for SegBBox2DList objects, allowing operations on collections of segmentation
        bounding boxes.

Type Variables:
    - T: A type variable representing the numeric type of the bounding box coordinates (int or float).
    - PT: A type variable representing the point type (Point2D[int] or Point2D[float]).
    - SBBT:
        A type variable representing the segmentation bounding box type
        (SegBBox2D[Point2D[int]] or SegBBox2D[Point2D[float]]).
    - SBBLT:
        A type variable representing the segmentation bounding box list type
        (SegBBox2DList[SegBBox2D[Point2D[int]]] or SegBBox2DList[SegBBox2D[Point2D[float]]]).

Type Aliases:
    - AnySegBBox2D:
        A type alias representing any segmentation bounding box type
        (SegBBox2D[Point2D[int]] or SegBBox2D[Point2D[float]]).
    - AnySegBBox2D:
        A type alias representing any segmentation bounding box type
        (SegBBox2D[Point2D[int]] or SegBBox2D[Point2D[float]]).
    - IntSegBBox2D:
        A type alias representing a segmentation bounding box with integer coordinates
        (SegBBox2D[Point2D[int]]).
    - FloatSegBBox2D:
        A type alias representing a segmentation bounding box with float coordinates
        (SegBBox2D[Point2D[float]]).
    - AnySegBBox2DList:
        A type alias representing any segmentation bounding box list type
        (SegBBox2DList[SegBBox2D[Point2D[int]]] or SegBBox2DList[SegBBox2D[Point2D[float]]]).
    - IntSegBBox2DList:
        A type alias representing a segmentation bounding box list with integer coordinates
        (SegBBox2DList[SegBBox2D[Point2D[int]]]).
    - FloatSegBBox2DList:
        A type alias representing a segmentation bounding box list with float coordinates
        (SegBBox2DList[SegBBox2D[Point2D[float]]]).
    - AnySegBBox2DNestedList:
        A type alias representing any segmentation bounding box nested list type
        (SegBBox2DNestedList[SegBBox2DList[SegBBox2D[Point2D[int]]]] or
         SegBBox2DNestedList[SegBBox2DList[SegBBox2D[Point2D[float]]]]).
    - IntSegBBox2DNestedList:
        A type alias representing a segmentation bounding box nested list with integer coordinates
        (SegBBox2DNestedList[SegBBox2DList[SegBBox2D[Point2D[int]]]]).
    - FloatSegBBox2DNestedList:
        A type alias representing a segmentation bounding box nested list with float coordinates
        (SegBBox2DNestedList[SegBBox2DList[SegBBox2D[Point2D[float]]]]).
"""

from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

import cv2
import numpy as np
import numpy.typing as npt

from aib.cv.geom.box.bbox2d import BBox2D, BBox2DList, BBox2DNestedList
from aib.cv.geom.box.utils.coord_formats import cxyar_to_xyxy, cxywh_to_xyxy, xywh_to_xyxy
from aib.cv.geom.point.point2d import AnyPoint2D, FloatPoint2D, IntPoint2D, Point2D
from aib.cv.geom.size import Size
from aib.cv.img.image import Image2D

if TYPE_CHECKING:
    AnySegBBox2D: TypeAlias = Union["SegBBox2D[IntPoint2D]", "SegBBox2D[FloatPoint2D]"]
    IntSegBBox2D: TypeAlias = "SegBBox2D[IntPoint2D]"
    FloatSegBBox2D: TypeAlias = "SegBBox2D[FloatPoint2D]"
    AnySegBBox2DList: TypeAlias = Union["SegBBox2DList[IntSegBBox2D]", "SegBBox2DList[FloatSegBBox2D]"]
    IntSegBBox2DList: TypeAlias = "SegBBox2DList[IntSegBBox2D]"
    FloatSegBBox2DList: TypeAlias = "SegBBox2DList[FloatSegBBox2D]"
    AnySegBBox2DNestedList: TypeAlias = Union[
        "SegBBox2DNestedList[IntSegBBox2DList]", "SegBBox2DNestedList[FloatSegBBox2DList]"
    ]
    IntSegBBox2DNestedList: TypeAlias = "SegBBox2DNestedList[IntSegBBox2DList]"
    FloatSegBBox2DNestedList: TypeAlias = "SegBBox2DNestedList[FloatSegBBox2DList]"
else:
    AnySegBBox2D = Union["SegBBox2D[IntPoint2D]", "SegBBox2D[FloatPoint2D]"]
    IntSegBBox2D = "SegBBox2D[IntPoint2D]"
    FloatSegBBox2D = "SegBBox2D[FloatPoint2D]"
    AnySegBBox2DList = Union["SegBBox2DList[IntSegBBox2D]", "SegBBox2DList[FloatSegBBox2D]"]
    IntSegBBox2DList = "SegBBox2DList[IntSegBBox2D]"
    FloatSegBBox2DList = "SegBBox2DList[FloatSegBBox2D]"
    AnySegBBox2DNestedList = Union["SegBBox2DNestedList[IntSegBBox2DList]", "SegBBox2DNestedList[FloatSegBBox2DList]"]
    IntSegBBox2DNestedList = "SegBBox2DNestedList[IntSegBBox2DList]"
    FloatSegBBox2DNestedList = "SegBBox2DNestedList[FloatSegBBox2DList]"

T = TypeVar("T", bound=Union[int, float], default=float)
PT = TypeVar("PT", bound=AnyPoint2D, default=FloatPoint2D)
SBBT = TypeVar("SBBT", bound=AnySegBBox2D, default=FloatSegBBox2D)
SBBLT = TypeVar("SBBLT", bound=AnySegBBox2DList, default=FloatSegBBox2DList)


@dataclass(frozen=True)
class SegBBox2D(BBox2D[PT]):
    """SegBBox2D Data Class

    Represents a 2D bounding box with segmentation mask.

    Attributes:
        p1 (Point2D): The top-left corner of the bounding box.
        p2 (Point2D): The bottom-right corner of the bounding box.
        score (float): The confidence score of the bounding box.
        label (int): The class label of the bounding box.
        mask (Image2D): The segmentation mask for the bounding box.
    """

    mask: Image2D = field(compare=False)

    def clamp(self, a_size: Size[int] | Size[float]) -> Self:
        """Clamp the bounding box coordinates to fit within the given size.

        Args:
            a_size (Size[int | float]): The size to clamp the bounding box to.

        Returns:
            SegBBox2D: A new SegBBox2D instance with clamped coordinates and resized mask.
        """
        p1 = Point2D(x=max(self.p1.x, type(self.p1.x)(0)), y=max(self.p1.y, type(self.p1.y)(0)))
        p2 = Point2D(x=min(self.p2.x, type(self.p2.x)(a_size.width)), y=min(self.p2.y, type(self.p2.y)(a_size.height)))
        mask = Image2D(
            cv2.resize(self.mask.data, (int(p2.x - p1.x), int(p2.y - p1.y)), interpolation=cv2.INTER_NEAREST)
        )
        return replace(self, p1=p1, p2=p2, mask=mask)

    def scale(self, a_factor: float, a_by_area: bool = False) -> Self:
        """Scale the bounding box by a factor, optionally by area.

        Args:
            a_factor (float): The scaling factor.
            a_by_area (bool): If True, scale by area (sqrt of factor), otherwise scale by width/height.

        Returns:
            SegBBox2D: A new SegBBox2D instance with scaled coordinates and mask.
        """
        if not isinstance(a_factor, (float, np.floating)):
            raise TypeError(f"a_factor must be a float, but got {type(a_factor)}.")
        if not 0 <= a_factor:
            raise ValueError(f"a_factor must be bigger than 0, but got {a_factor}.")
        if a_by_area:
            a_factor = np.sqrt(a_factor)
        width = self.width * a_factor
        height = self.height * a_factor
        p1 = Point2D(x=type(self.p1.x)(self.center.x - width / 2), y=type(self.p1.y)(self.center.y - height / 2))
        p2 = Point2D(x=type(self.p2.x)(self.center.x + width / 2), y=type(self.p2.y)(self.center.y + height / 2))
        mask = Image2D(
            cv2.resize(self.mask.data, (int(p2.x - p1.x), int(p2.y - p1.y)), interpolation=cv2.INTER_NEAREST)
        )
        return replace(self, p1=p1, p2=p2, mask=mask)

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> FloatSegBBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> IntSegBBox2D: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D:
        """Create a SegBBox2D instance from coordinates in various formats.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in xyxy format.
            a_coord_format (Literal['xyxy', 'xywh', 'cxywh', 'cxyar']):
                The format of the coordinates. Defaults to "xyxy".
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.
            a_mask (npt.NDArray[Any]): The segmentation mask.

        Returns:
            SegBBox2D: A SegBBox2D instance created from the provided coordinates.
        """
        if a_coord_format == "xyxy":
            return cls.from_xyxy(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, a_mask=a_mask, **kwargs
            )
        if a_coord_format == "xywh":
            return cls.from_xywh(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, a_mask=a_mask, **kwargs
            )
        if a_coord_format == "cxywh":
            return cls.from_cxywh(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, a_mask=a_mask, **kwargs
            )
        if a_coord_format == "cxyar":
            return cls.from_cxyar(
                a_coords=a_coords, a_use_float=a_use_float, a_score=a_score, a_label=a_label, a_mask=a_mask, **kwargs
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
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> FloatSegBBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> IntSegBBox2D: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D:
        """Create a SegBBox2D from xyxy coordinates and a mask.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.
            a_mask (npt.NDArray[Any]): The segmentation mask.

        Returns:
            SegBBox2D: A SegBBox2D instance created from the provided coordinates and mask.
        """
        p1 = Point2D.from_xy(a_coords[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(a_coords[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, mask=Image2D(a_mask), **kwargs)
        return cast(AnySegBBox2D, box)

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> FloatSegBBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> IntSegBBox2D: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D:
        """Create a SegBBox2D from xyxy coordinates and a mask.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.
            a_mask (npt.NDArray[Any]): The segmentation mask.

        Returns:
            SegBBox2D: A SegBBox2D instance created from the provided coordinates and mask.
        """
        xyxy = xywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, mask=Image2D(a_mask), **kwargs)
        return cast(AnySegBBox2D, box)

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> FloatSegBBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> IntSegBBox2D: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D:
        """Create a SegBBox2D from cxyar coordinates and a mask.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.
            a_mask (npt.NDArray[Any]): The segmentation mask.

        Returns:
            SegBBox2D: A SegBBox2D instance created from the provided coordinates and mask.
        """
        xyxy = cxyar_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, mask=Image2D(a_mask), **kwargs)
        return cast(AnySegBBox2D, box)

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> FloatSegBBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> IntSegBBox2D: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: None = None,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[float | int] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_score: float = 0.0,
        a_label: int = 0,
        a_mask: npt.NDArray[Any] = np.empty((0, 0)),
        **kwargs: Any,
    ) -> AnySegBBox2D:
        """Create a SegBBox2D from cxywh coordinates and a mask.

        Args:
            a_coords (Sequence[float | int] | npt.NDArray[np.float32]): The coordinates in xyxy format.
            a_use_float (Optional[bool]):
                If True, use Point2D[float]; otherwise, use Point2D[int]. If None, inferred from coords.
            a_score (float):
                The confidence score for the bounding box. Defaults to 0.0. If not provided, will be extracted
                from coords.
            a_label (int):
                The label for the bounding box. Defaults to 0. If not provided, will be extracted from coords.
            a_mask (npt.NDArray[Any]): The segmentation mask.

        Returns:
            SegBBox2D: A SegBBox2D instance created from the provided coordinates and mask.
        """
        xyxy = cxywh_to_xyxy(*a_coords[:4])
        p1 = Point2D.from_xy(xyxy[:2], a_use_float=a_use_float)
        p2 = Point2D.from_xy(xyxy[2:], a_use_float=a_use_float)
        score = a_score if len(a_coords) <= 4 else float(a_coords[4])
        label = a_label if len(a_coords) <= 5 else int(a_coords[5])
        box = cls(p1=cast(PT, p1), p2=cast(PT, p2), score=score, label=label, mask=Image2D(a_mask), **kwargs)
        return cast(AnySegBBox2D, box)

    def __iter__(self) -> Iterable[Any]:
        """Iterate over the SegBBox2D attributes.

        Yields:
            Iterable[Any]: The x1, y1, x2, y2 coordinates, score, label, and mask.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y
        yield self.score
        yield self.label
        yield self.mask

    def __getitem__(self, a_index: int) -> Any:
        """Get the value at the specified index.

        Args:
            a_index (int): The index to access.

        Returns:
            Any: The value at the specified index.

        Raises:
            IndexError: If the index is out of range.
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
        if a_index == 6:
            return self.mask
        raise IndexError(
            f"Invalid index {a_index}: A `SegBBox2D` object only supports indices "
            f"0 (x1), 1 (y1), 2 (x2), 3 (y2), 4(score), 5(label), 6(mask)."
        )

    def to_bbox2d(self) -> BBox2D[PT]:
        """Convert the SegBBox2D to a BBox2D without the mask.

        Returns:
            BBox2D: A BBox2D instance with the same coordinates, score, and label.
        """
        return BBox2D(p1=self.p1, p2=self.p2, score=self.score, label=self.label)


class SegBBox2DList(BBox2DList[SBBT]):
    """SegBBox2DList Data Container Class

    A list-like container for SegBBox2D objects, allowing for operations on collections of segmentation bounding boxes.
    Supports initialization from an iterable of SegBBox2D objects and conversion to NumPy arrays.

    Attributes:
        data (List[SBBT]): The list of SegBBox2D objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[SBBT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "SegBBox2DList",
    ):
        """Initialize a SegBBox2DList.

        Args:
            a_iterable (Optional[Iterable[SBBT]]): An optional iterable to initialize the list.
            a_max_size (Optional[int]): An optional maximum size for the list.
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
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> FloatSegBBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> IntSegBBox2DList: ...

    @overload
    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'],
        a_use_float: None = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList: ...

    @classmethod
    def create(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_coord_format: Literal['xyxy', 'xywh', 'cxywh', 'cxyar'] = "xyxy",
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList:
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
                a_coords=a_coords,
                a_use_float=a_use_float,
                a_scores=a_scores,
                a_labels=a_labels,
                a_masks=a_masks,
                **kwargs,
            )
        if a_coord_format == "xywh":
            return cls.from_xywh(
                a_coords=a_coords,
                a_use_float=a_use_float,
                a_scores=a_scores,
                a_labels=a_labels,
                a_masks=a_masks,
                **kwargs,
            )
        if a_coord_format == "cxywh":
            return cls.from_cxywh(
                a_coords=a_coords,
                a_use_float=a_use_float,
                a_scores=a_scores,
                a_labels=a_labels,
                a_masks=a_masks,
                **kwargs,
            )
        if a_coord_format == "cxyar":
            return cls.from_cxyar(
                a_coords=a_coords,
                a_use_float=a_use_float,
                a_scores=a_scores,
                a_labels=a_labels,
                a_masks=a_masks,
                **kwargs,
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
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> FloatSegBBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> IntSegBBox2DList: ...

    @overload
    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList: ...

    @classmethod
    def from_xyxy(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList:
        """Create a SegBBox2DList from xyxy coordinates and masks.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing coordinates in xyxy format.
            a_masks (npt.NDArray[Any]): The segmentation masks corresponding to the coordinates.
            a_point_type (Optional[type]):
                The Point2D type to use (Point2D[int] or Point2D[float]). If None, inferred from coords.

        Returns:
            SegBBox2DList: A SegBBox2DList instance created from the provided coordinates and masks.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        masks_list: List[npt.NDArray[Any]] | None = [mask for mask in a_masks] if a_masks is not None else None
        bboxes: List[AnySegBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            mask = masks_list[i] if masks_list and i < len(masks_list) else np.empty((0, 0))
            bbox = SegBBox2D.from_xyxy(
                a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, a_mask=mask, **kwargs
            )
            bboxes.append(bbox)
        return cast(AnySegBBox2DList, cls(cast(Iterable[SBBT], bboxes)))

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> FloatSegBBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> IntSegBBox2DList: ...

    @overload
    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList: ...

    @classmethod
    def from_xywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList:
        """Create a SegBBox2DList from xywh coordinates and masks.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing coordinates in xywh format.
            a_masks (npt.NDArray[Any]): The segmentation masks corresponding to the coordinates.
            a_point_type (Optional[type]):
                The Point2D type to use (Point2D[int] or Point2D[float]). If None, inferred from coords.

        Returns:
            SegBBox2DList: A SegBBox2DList instance created from the provided coordinates and masks.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        masks_list: List[npt.NDArray[Any]] | None = [mask for mask in a_masks] if a_masks is not None else None
        bboxes: List[AnySegBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            mask = masks_list[i] if masks_list and i < len(masks_list) else np.empty((0, 0))
            bbox = SegBBox2D.from_xywh(
                a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, a_mask=mask, **kwargs
            )
            bboxes.append(bbox)
        return cast(AnySegBBox2DList, cls(cast(Iterable[SBBT], bboxes)))

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> FloatSegBBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> IntSegBBox2DList: ...

    @overload
    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList: ...

    @classmethod
    def from_cxyar(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList:
        """Create a SegBBox2DList from cxyar coordinates and masks.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing coordinates in cxyar format.
            a_masks (npt.NDArray[Any]): The segmentation masks corresponding to the coordinates.
            a_point_type (Optional[type]):
                The Point2D type to use (Point2D[int] or Point2D[float]). If None, inferred from coords.

        Returns:
            SegBBox2DList: A SegBBox2DList instance created from the provided coordinates and masks.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        masks_list: List[npt.NDArray[Any]] | None = [mask for mask in a_masks] if a_masks is not None else None
        bboxes: List[AnySegBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            mask = masks_list[i] if masks_list and i < len(masks_list) else np.empty((0, 0))
            bbox = SegBBox2D.from_cxyar(
                a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, a_mask=mask, **kwargs
            )
            bboxes.append(bbox)
        return cast(AnySegBBox2DList, cls(cast(Iterable[SBBT], bboxes)))

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[True],
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> FloatSegBBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Literal[False] = False,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> IntSegBBox2DList: ...

    @overload
    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = None,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList: ...

    @classmethod
    def from_cxywh(
        cls,
        a_coords: Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer],
        a_use_float: Optional[bool] = True,
        a_scores: Optional[Sequence[float] | npt.NDArray[np.floating]] = None,
        a_labels: Optional[Sequence[int] | npt.NDArray[np.integer]] = None,
        a_masks: Optional[Sequence[npt.NDArray[Any]] | npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> AnySegBBox2DList:
        """Create a SegBBox2DList from cxywh coordinates and masks.

        Args:
            a_coords (Sequence[Sequence[float | int]] | npt.NDArray[np.floating | np.integer]):
                A sequence or NumPy array containing coordinates in cxywh format.
            a_masks (npt.NDArray[Any]): The segmentation masks corresponding to the coordinates.
            a_point_type (Optional[type]):
                The Point2D type to use (Point2D[int] or Point2D[float]). If None, inferred from coords.

        Returns:
            SegBBox2DList: A SegBBox2DList instance created from the provided coordinates and masks.
        """
        if isinstance(a_coords, np.ndarray) and a_coords.ndim == 1:
            a_coords = a_coords[np.newaxis]
        coords_list: List[List[float | int]] = [list(coord) for coord in a_coords]
        scores_list: List[float] | None = [float(score) for score in a_scores] if a_scores is not None else None
        labels_list: List[int] | None = [int(label) for label in a_labels] if a_labels is not None else None
        masks_list: List[npt.NDArray[Any]] | None = [mask for mask in a_masks] if a_masks is not None else None
        bboxes: List[AnySegBBox2D] = []
        for i, coord in enumerate(coords_list):
            score = scores_list[i] if scores_list and i < len(scores_list) else 0.0
            label = labels_list[i] if labels_list and i < len(labels_list) else 0
            mask = masks_list[i] if masks_list and i < len(masks_list) else np.empty((0, 0))
            bbox = SegBBox2D.from_cxywh(
                a_coords=coord, a_use_float=a_use_float, a_score=score, a_label=label, a_mask=mask, **kwargs
            )
            bboxes.append(bbox)
        return cast(AnySegBBox2DList, cls(cast(Iterable[SBBT], bboxes)))


class SegBBox2DNestedList(BBox2DNestedList[SBBLT]):
    """SegBBox2DNestedList Data Container Class

    A nested list-like container for SegBBox2DList objects, allowing for operations on collections of segmentation
    bounding boxes. Supports initialization from an iterable of SegBBox2DList objects and conversion to a flat
    SegBBox2DList.

    Attributes:
        data (List[SegBBox2DList]): The list of SegBBox2DList objects.
    """

    def __init__(
        self,
        a_iterable: Optional[Iterable[SBBLT]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "SegBBox2DNestedList",
    ):
        """Initialize a SegBBox2DNestedList.

        Args:
            a_iterable (Optional[Iterable[SBBLT]]): An optional iterable to initialize the list.
            a_max_size (Optional[int]): An optional maximum size for the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)

    def to_segbbox2dlist(self) -> SBBLT:
        """Convert the nested list of SegBBox2DList to a flat SegBBox2DList.

        Returns:
            SBBLT: A flat SegBBox2DList containing all SegBBox2D objects from the nested lists.
        """
        boxes: SegBBox2DList[SegBBox2D[Point2D[float]] | SegBBox2D[Point2D[int]]] = SegBBox2DList()
        for item in self:
            for box in item:
                boxes.append(box)
        return cast(SBBLT, boxes)


if not TYPE_CHECKING:
    IntSegBBox2D = SegBBox2D[IntPoint2D]
    FloatSegBBox2D = SegBBox2D[FloatPoint2D]
    IntSegBBox2DList = SegBBox2DList[IntSegBBox2D]
    FloatSegBBox2DList = SegBBox2DList[FloatSegBBox2D]
    IntSegBBox2DNestedList = SegBBox2DNestedList[IntSegBBox2DList]
    FloatSegBBox2DNestedList = SegBBox2DNestedList[FloatSegBBox2DList]
