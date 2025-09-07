"""
Segmented Instance Module

This module defines classes related to segmented instances in 2D space.

Classes:
    SegBBox2D: Represents a segmented bounding box in 2D space, extending the BBox2D class with an additional mask.
    SegBBox2DList: A container for managing a collection of SegBBox2D objects.
"""

# region Import Dependencies
from typing import List, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.cv.shape.bx import (
    BBox2D,
    BBox2DList,
    xywh_to_xyxy,
    cxyar_to_xyxy,
    cxywh_to_xyxy,
    BBox2DNestedList,
)
from aib.cv.shape.pt import Point2D
from aib.misc import is_int
from aib.obj import BaseObjectList


# endregion Import Dependencies


class SegBBox2D(BBox2D):
    """2D Segmented Bounding Box

    This class extends the functionality of BBox2D to include segmentation masks and labels.

    Attributes:
        mask (Image2D): The segmentation mask associated with the bounding box.
        label (int): The label associated with the bounding box.
    """

    def __init__(
        self,
        a_p1: Point2D,
        a_p2: Point2D,
        a_score: float,
        a_mask: Image2D,
        a_label: int,
        a_img_size: Optional[Size] = None,
        a_strict: Optional[bool] = False,
        a_conf_thre: Optional[float] = None,
        a_min_size_thre: Optional[Size] = None,
        a_do_validate: Optional[bool] = True,
        a_name: str = "SegBBox2D",
    ):
        """Initialize a SegBBox2D instance.

        Args:
            a_p1 (Point2D): The top-left point of the bounding box.
            a_p2 (Point2D): The bottom-right point of the bounding box.
            a_score (float): The score associated with the bounding box.
            a_mask (Image2D): The segmentation mask associated with the bounding box.
            a_label (int): The label associated with the bounding box.
            a_img_size (Optional[Size], optional): The size of the image containing the bounding box. Defaults to None.
            a_strict (Optional[bool], optional):
                A flag indicating whether strict validation should be applied. Defaults to False.
            a_conf_thre (Optional[float], optional): The confidence threshold for the bounding box. Defaults to None.
            a_min_size_thre (Optional[Size], optional):
                The minimum size threshold for the bounding box. Defaults to None.
            a_do_validate (Optional[bool], optional):
                A flag indicating whether validation should be performed. Defaults to True.
            a_name (str, optional): The name of the SegBBox2D instance. Defaults to "SegBBox2D".

        Raises:
            TypeError: If the label or mask is not of the correct type.
        """
        super().__init__(
            a_p1=a_p1,
            a_p2=a_p2,
            a_score=a_score,
            a_img_size=a_img_size,
            a_strict=a_strict,
            a_conf_thre=a_conf_thre,
            a_min_size_thre=a_min_size_thre,
            a_do_validate=a_do_validate,
            a_name=a_name,
        )
        self.mask: Image2D = a_mask
        self.label: int = a_label

    @property
    def label(self) -> int:
        """Get the label of the bounding box.

        Returns:
            int: The label of the bounding box.
        """
        return self._label

    @label.setter
    def label(self, a_label: int) -> None:
        """Set the label of the bounding box.

        Args:
            a_label (int): The label to be set.

        Raises:
            TypeError: If the label is not an integer.
        """
        if a_label is None or not is_int(a_label):
            raise TypeError("The `a_label` must be a `int`.")
        self._label: int = a_label

    @property
    def mask(self) -> Image2D:
        """Get the mask associated with the segmented instance box.

        Returns:
            Image2D: The mask associated with the segmented instance box.
        """
        return self._mask

    @mask.setter
    def mask(self, a_mask: Image2D) -> None:
        """Set the mask associated with the segmented instance box.

        Args:
            a_mask (Image2D): The mask to be associated with the segmented instance box.

        Raises:
            TypeError: If the mask is not an instance of Image2D or None.
        """
        if a_mask is None or not isinstance(a_mask, Image2D):
            raise TypeError("The `a_mask` must be a `Image2D`.")
        if self.do_validate and self.strict and a_mask.size != self.size:
            raise TypeError("The `a_mask` must be the same size as the bounding box.")
        self._mask: Image2D = a_mask

    @classmethod
    def validate_array(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_mask: npt.NDArray[np.floating | np.uint8],
    ) -> None:
        """Validate the coordinates array and mask array.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): Coordinates of the bounding box in [x, y, x, y, s] format.
            a_mask (npt.NDArray[np.floating]): Mask array.

        Raises:
            TypeError: If the coordinates array is not a tuple, list, or numpy array,
                or if the mask array is not a numpy array of type np.floating or None.
            ValueError: If the coordinates array has less than 5 elements, or if the mask array
                is not a 2D array, or if the coordinates array is multidimensional.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if a_mask is None or (
            not isinstance(a_mask, np.ndarray) and not np.issubdtype(a_mask.dtype, [np.floating, np.uint8])
        ):
            raise TypeError("The `a_mask` must be a `npt.NDArray[np.floating | np.uint8]`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 5:
            raise ValueError(
                f"`a_coordinates` array should have at least length 5 but it is in shape of" f" {a_coordinates.shape}."
            )

        if a_mask.ndim != 2:
            raise ValueError(f"`a_mask` array must be a 2D array but it has {a_mask.ndim} dimensions.")

        if a_coordinates.ndim > 1:
            raise ValueError(f"`a_coordinates` array should be a 1D array but it has {a_coordinates.ndim} dimensions.")

    @classmethod
    def from_xyxys(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_mask: npt.NDArray[np.floating | np.uint8],
        a_label: Union[int | np.integer],
        **kwargs,
    ) -> "SegBBox2D":
        """Create a SegBBox2D instance from coordinates in [x, y, x, y, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): Coordinates of the bounding boxes in [x, y, x, y, s] format.
            a_mask (npt.NDArray[np.floating]): Mask array.
            a_label (Union[int | np.integer]): Label for the bounding box.

        Returns:
            SegBBox2D: An instance of the SegBBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates, a_mask=a_mask)

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            a_mask=Image2D(a_mask),
            a_label=a_label,
            **kwargs,
        )
        return box

    @classmethod
    def from_xywhs(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_mask: npt.NDArray[np.floating | np.uint8],
        a_label: Union[int | np.integer],
        **kwargs,
    ) -> "SegBBox2D":
        """Create a SegBBox2D instance from coordinates in [x, y, w, h, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): Coordinates of the bounding boxes in [x, y, w, h, s] format.
            a_mask (npt.NDArray[np.floating]): Mask array.
            a_label (Union[int | np.integer]): Label for the bounding box.

        Returns:
            SegBBox2D: An instance of the SegBBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates, a_mask=a_mask)

        # Reformat xywh to xyxy
        a_coordinates[:4] = xywh_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            a_mask=Image2D(a_mask),
            a_label=a_label,
            **kwargs,
        )
        return box

    @classmethod
    def from_cxyars(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_mask: npt.NDArray[np.floating | np.uint8],
        a_label: Union[int | np.integer],
        **kwargs,
    ) -> "SegBBox2D":
        """Create a SegBBox2D instance from coordinates in [center-x, center-y, area, aspect_ratio, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): Coordinates of the bounding boxes in [center-x, center-y,
                area, aspect_ratio, s] format.
            a_mask (npt.NDArray[np.floating]): Mask array.
            a_label (Union[int | np.integer]): Label for the bounding box.

        Returns:
            SegBBox2D: An instance of the SegBBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates, a_mask=a_mask)

        # Reformat (center-x, center-y, area, aspect_ratio) to (x, y, x, y)
        a_coordinates[:4] = cxyar_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            a_mask=Image2D(a_mask),
            a_label=a_label,
            **kwargs,
        )
        return box

    @classmethod
    def from_cxywhs(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_mask: npt.NDArray[np.floating | np.uint8],
        a_label: Union[int | np.integer],
        **kwargs,
    ) -> "SegBBox2D":
        """Create a SegBBox2D instance from coordinates in [center-x, center-y, width, height, s] format.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]): Coordinates of the bounding boxes in [center-x, center-y,
                width, height, s] format.
            a_mask (npt.NDArray[np.floating]): Mask array.
            a_label (Union[int | np.integer]): Label for the bounding box.

        Returns:
            SegBBox2D: An instance of the SegBBox2D class.
        """
        # Validate array
        cls.validate_array(a_coordinates=a_coordinates, a_mask=a_mask)

        # Reformat (center-x, center-y, width, height) to (x, y, x, y)
        a_coordinates[:4] = cxywh_to_xyxy(*a_coordinates[:4])

        # Instantiate bbox
        box = cls(
            a_p1=Point2D(a_x=a_coordinates[0], a_y=a_coordinates[1]),
            a_p2=Point2D(a_x=a_coordinates[2], a_y=a_coordinates[3]),
            a_score=a_coordinates[4],
            a_mask=Image2D(a_mask),
            a_label=a_label,
            **kwargs,
        )
        return box

    def __iter__(self):
        """Iterate over the SegBBox2D object.

        Yields:
            float | int | Image2D: X and Y coordinates of points, score, mask, and label.
        """
        yield self.p1.x
        yield self.p1.y
        yield self.p2.x
        yield self.p2.y
        yield self.score
        yield self.mask
        yield self.label

    def __getitem__(self, index):
        """Get item from the SegBBox2D object.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            float | int | Image2D: X and Y coordinates of points, score, mask, and label.

        Raises:
            IndexError: If the index is out of range for the SegBBox2D object.
        """
        if index == 0:
            return self.p1.x
        elif index == 1:
            return self.p1.y
        elif index == 2:
            return self.p2.x
        elif index == 3:
            return self.p2.y
        elif index == 4:
            return self.score
        elif index == 5:
            return self.mask
        elif index == 6:
            return self.label
        else:
            raise IndexError("Index out of range for SegBBox2D object")

    # TODO(doc): Complete the document of following method
    def to_bbox2d(self) -> BBox2D:
        return BBox2D(
            a_p1=self.p1,
            a_p2=self.p2,
            a_score=self.score,
            a_img_size=self.img_size,
            a_strict=self.strict,
            a_conf_thre=self.conf_thre,
            a_min_size_thre=self.min_size_thre,
            a_do_validate=self.do_validate,
            a_name=self.name,
        )


class SegBBox2DList(BBox2DList, BaseObjectList[SegBBox2D]):
    """2D Segmented Bounding Box List

    The SegBBox2DList class is based on the BBox2DList class and serves as a container for a collection of Segmented
    Bounding Box (SegBBox2D) objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the SegBBox2DList (default is 'SegBBox2DList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[SegBBox2D], optional):
            A list of SegBBox2D objects to initialize the SegBBox2DList (default is None).
    """

    def __init__(
        self,
        a_name: str = "SegBBox2DList",
        a_max_size: int = -1,
        a_items: List[SegBBox2D] = None,
    ):
        """Initialize SegBBox2DList.

        Args:
            a_name (str, optional):
                The name of the list. Default is "SegBBox2DList".
            a_max_size (int, optional):
                The maximum size of the list. Default is -1, indicating no size limit.
            a_items (List[SegBBox2D], optional):
                Initial list of SegBBox2D items. Default is None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    @classmethod
    def validate_array(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_masks: npt.NDArray[np.floating | np.uint8],
        a_labels: npt.NDArray[np.integer],
        **kwargs,
    ) -> np.ndarray:
        """Validate coordinates, masks, and labels arrays.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding boxes in [x, y, x, y, s] format.
            a_masks (npt.NDArray[np.floating]):
                Masks associated with the bounding boxes.
            a_labels (npt.NDArray[np.integer]):
                Labels associated with the bounding boxes.
            **kwargs:
                Additional keyword arguments.

        Returns:
            np.ndarray: The validated coordinates as an array.

        Raises:
            TypeError: If the input arrays are not of the expected types.
            ValueError: If the input arrays do not have the expected shapes.
        """
        if a_coordinates is None and not isinstance(a_coordinates, (Tuple, List, np.ndarray)):
            raise TypeError("The `a_coordinates` should be a `Tuple, List, or np.ndarray`.")

        if a_masks is None or (
            not isinstance(a_masks, np.ndarray) and not np.issubdtype(a_masks.dtype, [np.floating, np.uint8])
        ):
            raise TypeError("The `a_mask` must be a `npt.NDArray[np.floating | np.uint8]`.")

        if a_labels is None and not isinstance(a_labels, np.ndarray) and not np.issubdtype(a_labels.dtype, np.integer):
            raise TypeError("The `a_label` must be a `npt.NDArray[np.integer]`.")

        if not isinstance(a_coordinates, np.ndarray):
            a_coordinates = np.array(a_coordinates)

        if a_coordinates.shape[-1] < 5:
            raise ValueError(
                f"`a_coordinates` array should have length 5 in the last dimension(-1) but it has "
                f"{a_coordinates.shape}."
            )

        if a_masks.ndim != 3:
            raise ValueError(f"`a_mask` array must be a batched 2D array but it has {a_masks.ndim} dimensions.")

        if a_coordinates.ndim == 1:
            a_coordinates = a_coordinates[np.newaxis]

        return a_coordinates

    @classmethod
    def from_xyxys(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_masks: npt.NDArray[np.floating | np.uint8],
        a_labels: npt.NDArray[np.integer],
        **kwargs,
    ) -> "SegBBox2DList":
        """XYXYS Initializer

        It creates a SegBBox2DList instance from coordinates in [x, y, x, y, s] format with corresponding masks and
        labels.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding boxes in [x, y, x, y, s] format.
            a_masks (npt.NDArray[np.floating]):
                Masks associated with the bounding boxes.
            a_labels (npt.NDArray[np.integer]):
                Labels associated with the bounding boxes.
            **kwargs:
                Additional keyword arguments to pass to the SegBBox2D constructor.

        Returns:
            SegBBox2DList: An instance of the SegBBox2DList class.
        """

        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates, a_masks=a_masks, a_labels=a_labels)

        # Instantiate bounding boxes
        bboxes = SegBBox2DList()
        bboxes.append(
            a_item=[
                SegBBox2D.from_xyxys(a_coordinates=coord, a_mask=mask, a_label=label, **kwargs)
                for coord, mask, label in zip(coordinates, a_masks, a_labels)
            ]
        )
        return bboxes

    @classmethod
    def from_xywhs(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_masks: npt.NDArray[np.floating | np.uint8],
        a_labels: npt.NDArray[np.integer],
        **kwargs,
    ) -> "SegBBox2DList":
        """XYWHS Initializer

        It creates a SegBBox2DList instance from coordinates in [x, y, w, h, s] format with corresponding masks and
        labels.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding boxes in [x, y, w, h, s] format.
            a_masks (npt.NDArray[np.floating]):
                Masks associated with the bounding boxes.
            a_labels (npt.NDArray[np.integer]):
                Labels associated with the bounding boxes.
            **kwargs:
                Additional keyword arguments to pass to the SegBBox2D constructor.

        Returns:
            SegBBox2DList: An instance of the SegBBox2DList class.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates, a_masks=a_masks, a_labels=a_labels)

        # Instantiate bounding boxes
        bboxes = SegBBox2DList()
        bboxes.append(
            a_item=[
                SegBBox2D.from_xywhs(a_coordinates=coord, a_mask=mask, a_label=label, **kwargs)
                for coord, mask, label in zip(coordinates, a_masks, a_labels)
            ]
        )
        return bboxes

    @classmethod
    def from_cxywhs(
        cls,
        a_coordinates: Union[Tuple, List, np.ndarray],
        a_masks: npt.NDArray[np.floating | np.uint8],
        a_labels: npt.NDArray[np.integer],
        **kwargs,
    ) -> "SegBBox2DList":
        """CXYWHS Initializer

        It creates a SegBBox2DList instance from coordinates in [center-x, center-y, width, height, s] format with
        corresponding masks and labels.

        Args:
            a_coordinates (Union[Tuple, List, np.ndarray]):
                Coordinates of the bounding boxes in [center-x, center-y, width, height, s] format.
            a_masks (npt.NDArray[np.floating]):
                Masks associated with the bounding boxes.
            a_labels (npt.NDArray[np.integer]):
                Labels associated with the bounding boxes.
            **kwargs:
                Additional keyword arguments to pass to the SegBBox2D constructor.

        Returns:
            SegBBox2DList: An instance of the SegBBox2DList class.
        """
        # Validate array
        coordinates = cls.validate_array(a_coordinates=a_coordinates, a_masks=a_masks, a_labels=a_labels)

        # Instantiate bounding boxes
        bboxes = SegBBox2DList()
        bboxes.append(
            a_item=[
                SegBBox2D.from_cxywhs(a_coordinates=coord, a_mask=mask, a_label=label, **kwargs)
                for coord, mask, label in zip(coordinates, a_masks, a_labels)
            ]
        )
        return bboxes


class SegBBox2DNestedList(BBox2DNestedList, BaseObjectList[SegBBox2DList]):
    """Batch of Segmented 2D Bounding Boxes Lists

    This class extends BatchBBox2DList and BaseObjectList to manage a collection of SegBBox2DList instances.
    It provides methods for managing and manipulating the list of grouped segmented bounding boxes.

    Attributes:
        Inherits attributes from :class:`BatchBBox2DList` and :class:`BaseObjectList`.
    """

    def __init__(
        self,
        a_name: str = "SegBBox2DNestedList",
        a_max_size: int = -1,
        a_items: List[SegBBox2DList] = None,
    ):
        """Initialize a SegBBox2DNestedList instance.

        Args:
            a_name (str, optional): The name of the SegBBox2DNestedList. Defaults to "SegBBox2DNestedList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1, indicating no size limit.
            a_items (List[SegBBox2DList], optional):
                A list of SegBBox2DList objects to initialize the SegBBox2DNestedList. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def to_segbbox2dlist(self):
        """Convert the SegBBox2DNestedList to a SegBBox2DList.

        This method consolidates all SegBBox2D objects from the contained SegBBox2DList instances into a single
        SegBBox2DList.

        Returns:
            SegBBox2DList: A consolidated list of SegBBox2D objects.
        """
        boxes = SegBBox2DList()
        for item in self.items:
            for box in item:
                boxes.append(box)
        return boxes
