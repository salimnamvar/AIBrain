"""Re-identification Target Modules
"""

# region Imported Dependencies
import uuid
from typing import List, Union, Optional, TypeVar, Generic
from aib.misc import Time
from aib.ml.reid.util.entity.desc import ReidDescList
from aib.ml.reid.util.entity.state import ReidState, TypeReidState
from aib.obj import ExtBaseObject, BaseObjectList, BaseObjectDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following type
TypeReidTarget = TypeVar("TypeReidTarget", bound="ReidTarget")


# TODO(doc): Complete the document of following class
class ReidTarget(Generic[TypeReidState], ExtBaseObject):
    def __init__(
        self,
        a_time: Time,
        a_id: Optional[uuid.UUID] = None,
        a_descriptors: Optional[ReidDescList] = None,
        a_num_desc: int = -1,
        a_state: Optional[ReidState] = None,
        a_state_type: type[ReidState] = ReidState,
        a_name: str = "ReidTarget",
    ):
        super().__init__(a_name=a_name, a_time=a_time)
        self.id: uuid.UUID = a_id
        self.descriptors: ReidDescList = (
            a_descriptors if a_descriptors is not None else ReidDescList(a_max_size=a_num_desc)
        )
        self._state_type: type[ReidState] = a_state_type
        self.state: ReidState = a_state if a_state else self._state_type(time=a_time)

    @property
    def state(self) -> TypeReidState:
        return self._state

    @state.setter
    def state(self, a_stats: ReidState) -> None:
        if a_stats is not None and not isinstance(a_stats, ReidState):
            raise TypeError(f"`a_stats` argument must be an `ReidState` but it's type is `{type(a_stats)}`")
        self._state: ReidState = a_stats

    def to_dict(self) -> dict:
        dic = {"name": self.name, "id": self.id, "time": self.time, "descriptors": self.descriptors.to_dict()}
        return dic

    def update(self, a_inst: TypeReidTarget) -> None:
        self.time = a_inst.time.copy()
        self.id = a_inst.id


# TODO(doc): Complete the document of following type
TypeReidTargetList = TypeVar("TypeReidTargetList", bound="ReidTargetList")


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


# TODO(doc): Complete the document of following type
TypeReidTargetDict = TypeVar("TypeReidTargetDict", bound="ReidTargetDict")


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
