"""Timestamp
"""

# region Imported Dependencies
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Union
from aib.obj import BaseObject, BaseObjectList

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class TimeDelta(BaseObject):
    def __init__(
        self, a_step: Optional[int] = None, a_timedelta: Optional[timedelta] = None, a_name: str = "TimeDelta"
    ):
        super().__init__(a_name=a_name)
        self.timedelta: timedelta = a_timedelta
        self.step: int = a_step

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, a_step: Optional[int] = None) -> None:
        if a_step is not None and not isinstance(a_step, int):
            raise TypeError("The `a_step` should be a `int` or `None`.")
        self._step: int = -1 if a_step is None else a_step

    @property
    def timedelta(self) -> timedelta:
        return self._timedelta

    @timedelta.setter
    def timedelta(self, a_timedelta: Optional[timedelta] = None) -> None:
        if a_timedelta is None or not isinstance(a_timedelta, timedelta):
            raise TypeError("The `a_timedelta` should be a `timedelta` or `None`.")
        self._timedelta: timedelta = a_timedelta

    def to_dict(self) -> dict:
        dic = {
            "name": self.name,
            "step": self.step,
            "timedelta": self.timedelta,
        }
        return dic


# TODO(doc): Complete the document of following class
class TimeDeltaList(BaseObjectList[TimeDelta]):
    def __init__(
        self,
        a_name: str = "TimeDeltaList",
        a_max_size: int = -1,
        a_items: Union[TimeDelta, List[TimeDelta], "TimeDeltaList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class Time(BaseObject):
    def __init__(self, a_step: Optional[int] = None, a_timestamp: Optional[datetime] = None, a_name: str = "Time"):
        super().__init__(a_name=a_name)
        self.timestamp: datetime = a_timestamp
        self.step: int = a_step

    def update(self, a_time: Optional["Time"] = None) -> None:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        self.timestamp = a_time.timestamp
        self.step = a_time.step

    def increment(self) -> None:
        self.step += 1
        self.timestamp = None

    def to_dict(self) -> dict:
        dic = {
            "name": self.name,
            "step": self.step,
            "timestamp": self.timestamp,
        }
        return dic

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, a_step: Optional[int] = None) -> None:
        if a_step is not None and not isinstance(a_step, int):
            raise TypeError("The `a_step` should be a `int` or `None`.")
        self._step: int = -1 if a_step is None else a_step

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, a_timestamp: Optional[datetime] = None) -> None:
        if a_timestamp is not None and not isinstance(a_timestamp, datetime):
            raise TypeError("The `a_timestamp` should be a `datetime` or `None`.")
        self._timestamp: datetime = (
            datetime.now().astimezone(tz=timezone(timedelta(hours=0))) if a_timestamp is None else a_timestamp
        )

    def __eq__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return self.timestamp == a_time.timestamp and self.step == a_time.step

    def __ne__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return not self.__eq__(a_time=a_time)

    def __lt__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return self.timestamp < a_time.timestamp and self.step < a_time.step

    def __gt__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return self.timestamp > a_time.timestamp and self.step > a_time.step

    def __le__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return self.timestamp <= a_time.timestamp and self.step <= a_time.step

    def __ge__(self, a_time: "Time") -> bool:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return self.timestamp >= a_time.timestamp and self.step >= a_time.step

    def __sub__(self, a_time: "Time") -> TimeDelta:
        if a_time is None or not isinstance(a_time, Time):
            raise TypeError("The `a_time` should be a `Time`.")
        return TimeDelta(a_step=self.step - a_time.step, a_timedelta=self.timestamp - a_time.timestamp)


# TODO(doc): Complete the document of following class
class TimeList(BaseObjectList[Time]):
    def __init__(
        self,
        a_name: str = "TimeList",
        a_max_size: int = -1,
        a_items: Union[Time, List[Time], "TimeList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
