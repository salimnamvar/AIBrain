"""Re-identification Entity State Module
"""

# region Imported Dependencies
from datetime import datetime, timezone, timedelta
from typing import List, Union, Optional
from brain.util.obj import BaseObject, BaseObjectDict, BaseObjectList

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidEntityState(BaseObject):
    def __init__(self, a_timestamp: Optional[datetime] = None, a_name: str = "ReidEntityState") -> None:
        super().__init__(a_name)
        self.timestamp: datetime = (
            datetime.now().astimezone(tz=timezone(timedelta(hours=0))) if a_timestamp is None else a_timestamp
        )

    def to_dict(self) -> dict:
        dic = {"name": self.name, "time": self.timestamp}
        return dic


# TODO(doc): Complete the document of following class
class ReidEntityStateList(BaseObjectList[ReidEntityState]):
    def __init__(
        self,
        a_name: str = "ReidEntityStateList",
        a_max_size: int = -1,
        a_items: Union[ReidEntityState, List[ReidEntityState], "ReidEntityStateList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class ReidEntityStateDict(BaseObjectDict[int, ReidEntityState]):
    def __init__(
        self,
        a_name: str = "ReidEntityStateDict",
        a_max_size: int = -1,
        a_key: Union[int, List[int]] = None,
        a_value: Union[ReidEntityState, List[ReidEntityState]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
