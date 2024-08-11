"""Identified Entity

This module defines the `Entity` class, representing a re-identification entity object.
"""

# region Imported Dependencies
import uuid
from datetime import datetime
from typing import List, Union, Optional
from brain.util.ml.reid.util.entity.desc import ReidDescList
from brain.util.ml.reid.util.entity.state import ReidEntityStateList, ReidEntityState
from brain.util.ml.reid.util.entity.tgt import ReidTarget, ReidTargetList, ReidTargetDict, ReidTargetNestedList
from brain.util.obj import BaseObjectList, BaseObjectDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidEntity(ReidTarget):
    def __init__(
        self,
        a_id: uuid.UUID,
        a_state: ReidEntityState,
        a_timestamp: Optional[datetime] = None,
        a_num_state: int = -1,
        a_descriptors: Optional[ReidDescList] = None,
        a_num_desc: int = -1,
        a_name: str = "ReidEntity",
    ):
        super().__init__(
            a_timestamp=a_state.timestamp if a_timestamp is None else a_timestamp,
            a_id=a_id,
            a_descriptors=a_descriptors,
            a_num_desc=a_num_desc,
            a_name=a_name,
        )
        self.states: ReidEntityStateList = ReidEntityStateList(a_items=a_state, a_max_size=a_num_state)

    def to_dict(self) -> dict:
        dic = super().to_dict()
        dic.update(
            {
                "states": self.states,
            }
        )
        return dic

    @property
    def last_state(self) -> ReidEntityState:
        return self.states[-1]

    @property
    def first_state(self) -> ReidEntityState:
        return self.states[0]

    def update(self, a_state: ReidEntityState, a_descriptors: ReidDescList) -> None:
        # UPDATE descriptors
        for desc in a_descriptors:
            if desc.timestamp >= self.last_state.timestamp:
                self.descriptors.append(desc)

        # INSERT state
        self.states.append(a_state)


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
