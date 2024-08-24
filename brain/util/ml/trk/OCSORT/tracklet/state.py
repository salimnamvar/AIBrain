"""Target State Module

This module defines the `State` class for representing the state of trackable objects,
and the `StateDict` class for managing a dictionary of states.

"""

# region Imported Dependencies
from datetime import datetime, timezone, timedelta
from typing import List, Union, Optional
from brain.util.obj import ExtBaseObject, BaseObjectDict, BaseObjectList
from brain.util.cv.shape.bx import BBox2D

# endregion Imported Dependencies


class State(ExtBaseObject):
    """Class representing the state of a trackable object.

    Attributes:
        box (BBox2D): Bounding box associated with the state.
        timestamp (datetime): Timestamp of when the state was created.
    """

    def __init__(self, a_box: Optional[BBox2D] = None, a_name: str = "State") -> None:
        """Initialize a State instance.

        Args:
            a_box (BBox2D, optional): Bounding box associated with the state (default is `None`).
            a_name (str, optional): Name of the state (default is "State").

        Raises:
            TypeError: If the input bounding box is not an instance of `BBox2D`.
        """
        super().__init__(a_name)
        self._timestamp: datetime = datetime.now().astimezone(tz=timezone(timedelta(hours=0)))
        self.box: BBox2D = a_box

    def to_dict(self) -> dict:
        """Convert the State instance to a dictionary.

        Returns:
            dict: A dictionary representation of the State instance, including its attributes.
        """
        dic = {"name": self.name, "bbox": self.box.to_dict()}
        return dic

    @property
    def box(self) -> BBox2D:
        """Get the bounding box associated with the state.

        Returns:
            BBox2D: The bounding box associated with the state.
        """
        return self._box

    @box.setter
    def box(self, a_box: BBox2D) -> None:
        """Set the bounding box associated with the state.

        Args:
            a_box (BBox2D): Bounding box associated with the state.

        Raises:
            TypeError: If the input bounding box is not an instance of `BBox2D`.
        """
        if a_box is not None and not isinstance(a_box, BBox2D):
            raise TypeError("The `a_box` must be a `BBox2D`.")
        self._box: BBox2D = a_box

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the state.

        Returns:
            datetime: The timestamp of the state.
        """
        return self._timestamp


# TODO(doc): Complete the document of following class
class StateList(BaseObjectList[State]):
    def __init__(self, a_name: str = "StateList", a_max_size: int = -1, a_items: List[State] = None):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


class StateDict(BaseObjectDict[int, State]):
    """Dictionary class for managing a collection of states.

    This class inherits from :class:`BaseObjectDict` and is specialized for managing a dictionary of :class:`State`
    instances.
    """

    def __init__(
        self,
        a_name: str = "State_Dict",
        a_max_size: int = -1,
        a_key: Union[int, List[int]] = None,
        a_value: Union[State, List[State]] = None,
        a_key_type: Optional[type] = None,
        a_value_type: Optional[type] = None,
    ):
        """Initialize a StateDict instance.

        Args:
            a_name (str, optional): Name of the state dictionary (default is "State_Dict").
            a_max_size (int, optional): Maximum size of the state dictionary (default is -1, indicating no limit).
            a_key (Union[int, List[int]], optional): Initial key or keys for the state dictionary (default is None).
            a_value (Union[State, List[State]], optional):
                Initial value or values for the state dictionary (default is None).
            a_key_type (type, optional):
                The key data type that specifies the type of keys in the dictionary.
            a_value_type (type, optional):
                The value data type that specifies the type of values in the dictionary.
        """
        super().__init__(
            a_name=a_name,
            a_max_size=a_max_size,
            a_key=a_key,
            a_value=a_value,
            a_key_type=a_key_type,
            a_value_type=a_value_type,
        )
