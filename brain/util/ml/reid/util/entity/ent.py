"""Identified Entity
"""

# region Imported Dependencies
import uuid
from typing import List, Union, Optional, TypeVar, Generic
from brain.util.misc import Time, TimeDelta
from brain.util.ml.reid.util.entity.desc import ReidDescList
from brain.util.ml.reid.util.entity.state import ReidStateTable, ReidState, TypeReidState, TypeReidStateTable
from brain.util.ml.reid.util.entity.tgt import (
    ReidTarget,
    ReidTargetList,
    ReidTargetDict,
    ReidTargetNestedList,
)
from brain.util.obj import BaseObjectList, BaseObjectDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following type
TypeReidEntity = TypeVar("TypeReidEntity", bound="ReidEntity")


# TODO(doc): Complete the document of following class
class ReidEntity(Generic[TypeReidStateTable, TypeReidState], ReidTarget[TypeReidState]):
    def __init__(
        self,
        a_time: Time,
        a_id: uuid.UUID,
        a_state: Optional[ReidState] = None,
        a_state_type: type[ReidState] = ReidState,
        a_num_state: int = -1,
        a_state_table_type: type[ReidStateTable] = ReidStateTable,
        a_descriptors: Optional[ReidDescList] = None,
        a_num_desc: int = -1,
        a_desc_samp_rate: int = 0,
        a_name: str = "ReidEntity",
    ):
        super().__init__(
            a_time=a_time,
            a_id=a_id,
            a_descriptors=a_descriptors,
            a_num_desc=a_num_desc,
            a_name=a_name,
            a_state=a_state,
            a_state_type=a_state_type,
        )
        self.desc_samp_rate: int = a_desc_samp_rate
        self._state_table_type: type[ReidStateTable] = a_state_table_type
        self._states: TypeReidStateTable = self._state_table_type(a_data=self.state, a_max_rows=a_num_state)

    @property
    def states(self) -> ReidStateTable:
        return self._states

    def to_dict(self) -> dict:
        dic = super().to_dict()
        dic.update(
            {
                "age": self.age,
                "state": self.states.last_state,
            }
        )
        return dic

    @property
    def age(self) -> TimeDelta:
        return self.states.last_state.time - self.time

    def update_descriptors(self, a_inst: Union[ReidTarget, TypeReidEntity]) -> None:
        # UPDATE descriptors
        for desc in a_inst.descriptors:
            if len(self.descriptors) == 0 or (desc.time.step > self.states.last_state.time.step + self.desc_samp_rate):
                self.descriptors.append(desc)

    def update(self, a_inst: Union[ReidTarget, TypeReidEntity]) -> None:
        # UPDATE descriptors
        self.update_descriptors(a_inst=a_inst)
        # Append state
        self.states.append(a_data=a_inst.state)


# TODO(doc): Complete the document of following class
class ReidEntityList(ReidTargetList, BaseObjectList[ReidEntity]):
    def __init__(
        self,
        a_name: str = "ReidEntityList",
        a_max_size: int = -1,
        a_items: Union[ReidEntity, List[ReidEntity], "ReidEntityList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class ReidEntityNestedList(ReidTargetNestedList, BaseObjectList[ReidEntityList]):
    def __init__(
        self,
        a_name: str = "ReidEntityNestedList",
        a_max_size: int = -1,
        a_items: Union[ReidEntityList, List[ReidEntityList], "ReidEntityNestedList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following type
TypeReidEntityDict = TypeVar("TypeReidEntityDict", bound="ReidEntityDict")


# TODO(doc): Complete the document of following class
class ReidEntityDict(ReidTargetDict, BaseObjectDict[uuid.UUID, ReidEntity]):
    def __init__(
        self,
        a_name: str = "ReidEntityDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[ReidEntity, List[ReidEntity]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
