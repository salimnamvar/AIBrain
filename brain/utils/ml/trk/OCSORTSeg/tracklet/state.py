"""Target State Module

"""

# region Imported Dependencies
from typing import Optional, Union, List

from brain.utils.ml.seg import SegBBox2D
from brain.utils.ml.trk.OCSORT import State as BaseState, StateDict as BaseStateDict
from brain.utils.obj import BaseObjectDict


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class State(BaseState):
    def __init__(
        self, a_box: Optional[SegBBox2D] = None, a_name: str = "State"
    ) -> None:
        super().__init__(a_box=a_box, a_name=a_name)

    @property
    def box(self) -> SegBBox2D:
        return self._box

    @box.setter
    def box(self, a_box: SegBBox2D) -> None:
        if a_box is not None and not isinstance(a_box, SegBBox2D):
            raise TypeError("The `a_box` must be a `SegBBox2D`.")
        self._box: SegBBox2D = a_box


# TODO(doc): Complete the document of following class
class StateDict(BaseStateDict, BaseObjectDict[int, State]):
    def __init__(
        self,
        a_name: str = "StateDict",
        a_max_size: int = -1,
        a_key: Union[int, List[int]] = None,
        a_value: Union[State, List[State]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
