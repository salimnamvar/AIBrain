"""Target State Module

This module defines the `State` class for representing the state of trackable objects,
and the `StateDict` class for managing a dictionary of states.

"""

# region Imported Dependencies
from datetime import datetime, timezone, timedelta
from typing import List, Union, Optional
from brain.utils.obj import BaseObject, BaseObjectDict
from brain.utils.cv.shape.bx import BBox2D

# endregion Imported Dependencies


class State(BaseObject):
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
        self._timestamp: datetime = datetime.now().astimezone(
            tz=timezone(timedelta(hours=0))
        )
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
    ):
        """Initialize a StateDict instance.

        Args:
            a_name (str, optional): Name of the state dictionary (default is "State_Dict").
            a_max_size (int, optional): Maximum size of the state dictionary (default is -1, indicating no limit).
            a_key (Union[int, List[int]], optional): Initial key or keys for the state dictionary (default is None).
            a_value (Union[State, List[State]], optional):
                Initial value or values for the state dictionary (default is None).
        """
        super().__init__(a_name, a_max_size, a_key, a_value)
