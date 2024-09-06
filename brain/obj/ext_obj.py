"""Extended Base Object
"""

# region Imported Dependencies
from abc import ABC
from typing import Optional
from brain.misc.time import Time
from brain.obj import BaseObject

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ExtBaseObject(BaseObject, ABC):
    def __init__(
        self,
        a_name: str = "Object",
        a_time: Optional[Time] = None,
    ) -> None:
        super().__init__(a_name=a_name)
        self.time: Time = a_time

    @property
    def time(self) -> Time:
        return self._time

    @time.setter
    def time(self, a_time: Time) -> None:
        if a_time is not None and not isinstance(a_time, Time):
            raise TypeError(f"`a_time` argument must be a `Time` but it's type is `{type(a_time)}`")
        self._time: Time = a_time
