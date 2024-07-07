"""Tracked Object Module

This module defines classes related to tracked objects in the 2D space.
"""

# region Import Dependencies
import uuid
from datetime import datetime
from typing import Optional, List, Union

from brain.utils.cv.shape import Size
from brain.utils.cv.shape.bx import BBox2D, BBox2DList
from brain.utils.cv.shape.pt import Point2D
from brain.utils.obj import BaseObjectDict, BaseObjectList


# endregion Import Dependencies


class TrackedBBox2D(BBox2D):
    """TrackedBBox2D Class

    This class extends the BBox2D class and represents a 2D bounding box tracked by an object tracker.
    It includes information about the bounding box coordinates, confidence score, and additional parameters related to
    tracking context.

    Attributes:
        id (uuid.UUID): The unique identifier for the tracked object.
        timestamp (datetime): The timestamp indicating when the object was tracked.
    """

    def __init__(
        self,
        a_id: uuid.UUID,
        a_timestamp: Optional[datetime],
        a_p1: Point2D,
        a_p2: Point2D,
        a_score: float,
        a_img_size: Optional[Size] = None,
        a_strict: Optional[bool] = False,
        a_conf_thre: Optional[float] = None,
        a_min_size_thre: Optional[Size] = None,
        a_do_validate: Optional[bool] = True,
        a_name: str = "TrackedBBox2D",
    ):
        super().__init__(
            a_p1=a_p1,
            a_p2=a_p2,
            a_score=a_score,
            a_img_size=a_img_size,
            a_strict=a_strict,
            a_conf_thre=a_conf_thre,
            a_min_size_thre=a_min_size_thre,
            a_name=a_name,
            a_do_validate=a_do_validate,
        )
        self.id: uuid.UUID = a_id
        self.timestamp: datetime = a_timestamp

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier

        Returns:
            uuid.UUID: The unique identifier of the tracked 2D bounding box.
        """
        return self._id

    @id.setter
    def id(self, a_id: uuid.UUID) -> None:
        """Set the unique identifier of the tracked 2D bounding box.

        Args:
            a_id (uuid.UUID): The unique identifier to set.

        Raises:
            TypeError: If the provided ID is not a valid UUID.
        """
        if a_id is None or not isinstance(a_id, uuid.UUID):
            raise TypeError("The `a_id` must be a `uuid.UUID`.")
        self._id: uuid.UUID = a_id

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp

        Returns:
            datetime: The timestamp when the tracked 2D bounding box was tracked.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, a_timestamp: datetime) -> None:
        """Set the timestamp when the tracked 2D bounding box was tracked.

        Args:
            a_timestamp (datetime): The timestamp to set.

        Raises:
            TypeError: If the provided timestamp is not a valid datetime object.
        """
        if a_timestamp is None or not isinstance(a_timestamp, datetime):
            raise TypeError("The `a_timestamp` must be a `datetime`.")
        self._timestamp: datetime = a_timestamp


class TrackedBBox2DList(BBox2DList, BaseObjectList[TrackedBBox2D]):
    """Represents a list of 2D bounding boxes tracked by an object tracker.

    This class extends the BBox2DList and serves as a container for a collection of TrackedBBox2D objects.
    It provides methods for managing and manipulating the list of tracked 2D bounding boxes.

    Attributes:
        Inherits attributes from :class:`BBox2DList`.
    """

    def __init__(
        self,
        a_name: str = "TrackedBBox2DList",
        a_max_size: int = -1,
        a_items: List[TrackedBBox2D] = None,
    ):
        """Initialize a TrackedBBox2DList instance.

        Args:
            a_name (str, optional): The name of the TrackedBBox2DList. Defaults to "TrackedBBox2DList".
            a_max_size (int, optional): The maximum size of the list. Defaults to -1, indicating no size limit.
            a_items (List[TrackedBBox2D], optional):
                A list of TrackedBBox2D objects to initialize the TrackedBBox2DList. Defaults to None.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


class TrackedBBox2DDict(BaseObjectDict[uuid.UUID, TrackedBBox2D]):
    """A dictionary to store tracked bounding boxes identified by their unique IDs.

    This class extends the BaseObjectDict class and is specialized to store TrackedBBox2D objects.
    """

    def __init__(
        self,
        a_name: str = "TrackedBBox2DDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[TrackedBBox2D, List[TrackedBBox2D]] = None,
    ):
        """
        Constructor for TrackedBBox2DDict

        Args:
            a_name (str, optional): The name of the TrackedBBox2DDict (default is 'TrackedBBox2DDict').
            a_max_size (int, optional): The maximum size of the dictionary (default is -1, indicating no size limit).
            a_key (Union[uuid.UUID, List[uuid.UUID]], optional):
                A single UUID key or a list of UUID keys to initialize the dictionary.
            a_value (Union[TrackedBBox2D, List[TrackedBBox2D]], optional):
                A single TrackedBBox2D value or a list of TrackedBBox2D values to initialize the dictionary.
        """
        super().__init__(a_name, a_max_size, a_key, a_value)
