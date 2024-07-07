"""Trackable Kalman-Filter Target
"""

# region Imported Dependencies
import uuid
from typing import Optional, List, Union

import numpy as np

from brain.utils.ml.seg import SegBBox2D
from brain.utils.ml.trk.OCSORT import (
    KFTarget as BaseKFTarget,
    KFTargetList as BaseKFTargetList,
    KFTargetDict as BaseKFTargetDict,
)
from brain.utils.obj import BaseObjectList, BaseObjectDict
from .state import State, StateDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class KFTarget(BaseKFTarget):
    def __init__(
        self,
        a_state: State,
        a_num_st_thre: int,
        a_delta_time: Optional[int] = 3,
        a_id: Optional[uuid.UUID] = None,
        a_name: Optional[str] = "KFTarget",
    ):
        super().__init__(
            a_state=a_state,
            a_num_st_thre=a_num_st_thre,
            a_delta_time=a_delta_time,
            a_id=a_id,
            a_name=a_name,
        )

    @property
    def states(self) -> StateDict:
        return self._states

    @property
    def state(self) -> State:
        return self._state

    @property
    def kf_state(self) -> State:
        coordinates = self._kf.x.squeeze().copy()
        box = None
        if np.any(coordinates[[2, 3]] < 0):
            state = State(a_box=box)
        else:
            coordinates = np.append(coordinates, self.state.box.score)
            state = State(
                a_box=SegBBox2D.from_cxyars(
                    a_coordinates=coordinates,
                    a_mask=self.state.box.mask.data,
                    a_label=self.state.box.label,
                    a_img_size=self.state.box.img_size,
                    a_strict=self.state.box.strict,
                    a_conf_thre=self.state.box.conf_thre,
                    a_min_size_thre=self.state.box.min_size_thre,
                    a_name=self._state.name,
                    a_do_validate=False,
                )
            )
        return state

    def predict(self) -> State:
        state = super().predict()
        return state

    def update(self, a_state: Optional[State] = None):
        super().update(a_state=a_state)


# TODO(doc): Complete the document of following class
class KFTargetList(BaseKFTargetList, BaseObjectList[KFTarget]):
    def __init__(
        self,
        a_name: str = "KFTargetList",
        a_max_size: int = -1,
        a_items: List[KFTarget] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class KFTargetDict(BaseKFTargetDict, BaseObjectDict[uuid.UUID, KFTarget]):
    def __init__(
        self,
        a_name: str = "KFTargetDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[KFTarget, List[KFTarget]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
