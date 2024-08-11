"""Re-identification Target Modules
"""

# region Imported Dependencies
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Union, Optional
from brain.util.ml.reid.util.entity.desc import ReidDescList
from brain.util.obj import BaseObject, BaseObjectList, BaseObjectDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidTarget(BaseObject):
    def __init__(
        self,
        a_id: Optional[uuid.UUID] = None,
        a_timestamp: Optional[datetime] = None,
        a_descriptors: Optional[ReidDescList] = None,
        a_num_desc: int = -1,
        a_name: str = "ReidTarget",
    ):
        super().__init__(a_name=a_name)
        self.timestamp: datetime = (
            datetime.now().astimezone(tz=timezone(timedelta(hours=0))) if a_timestamp is None else a_timestamp
        )
        self.id: uuid.UUID = a_id
        self.descriptors: ReidDescList = (
            a_descriptors if a_descriptors is not None else ReidDescList(a_max_size=a_num_desc)
        )

    def to_dict(self) -> dict:
        dic = {"name": self.name, "id": self.id, "timestamp": self.timestamp, "descriptors": self.descriptors.to_dict()}
        return dic

    def update(self, a_timestamp: datetime, a_id: uuid.UUID) -> None:
        self.timestamp = a_timestamp
        self.id = a_id


# TODO(doc): Complete the document of following class
class ReidTargetList(BaseObjectList[ReidTarget]):
    def __init__(
        self,
        a_name: str = "ReidTargetList",
        a_max_size: int = -1,
        a_items: Union[ReidTarget, List[ReidTarget], "ReidTargetList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class ReidTargetNestedList(BaseObjectList[ReidTargetList]):
    def __init__(
        self,
        a_name: str = "ReidTargetNestedList",
        a_max_size: int = -1,
        a_items: Union[ReidTargetList, List[ReidTargetList], "ReidTargetNestedList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class ReidTargetDict(BaseObjectDict[uuid.UUID, ReidTarget]):
    def __init__(
        self,
        a_name: str = "ReidTargetDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[ReidTarget, List[ReidTarget]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
