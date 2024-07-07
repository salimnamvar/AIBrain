"""Object Re-identification Descriptor Module.
"""

# region Imported Dependencies
from typing import List
import numpy as np
import numpy.typing as npt

from brain.utils.obj import BaseObject, BaseObjectList

# region Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidDesc(BaseObject):
    def __init__(
        self, a_features: npt.NDArray[np.floating], a_name: str = "ReidDesc"
    ) -> None:
        super().__init__(a_name=a_name)
        self._features: np.ndarray = a_features

    @property
    def features(self) -> np.ndarray:
        return self._features

    def to_dict(self) -> dict:
        dic = {"features": self.features}
        return dic


# TODO(doc): Complete the document of following class
class ReidDescList(BaseObjectList[ReidDesc]):
    def __init__(
        self,
        a_name: str = "ReidDescList",
        a_max_size: int = -1,
        a_items: List[ReidDesc] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
