"""Object Descriptor Base Module.
"""

# region Imported Dependencies
from typing import List, Union
import numpy as np
import numpy.typing as npt
from brain.misc import Time
from brain.obj import ExtBaseObject, BaseObjectList

# region Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidDesc(ExtBaseObject):
    def __init__(
        self, a_time: Time, a_features: npt.NDArray[np.floating], a_extractor: str = "FeatExt", a_name: str = "Desc"
    ) -> None:
        super().__init__(a_name=a_name, a_time=a_time)
        self._extractor: str = a_extractor
        self._features: np.ndarray = a_features

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def extractor(self) -> str:
        return self._extractor

    def to_dict(self) -> dict:
        dic = {"name": self.name, "extractor": self.extractor, "features": self.features, "time": self.time}
        return dic


# TODO(doc): Complete the document of following class
class ReidDescList(BaseObjectList[ReidDesc]):
    def __init__(
        self,
        a_name: str = "DescList",
        a_max_size: int = -1,
        a_items: Union[ReidDesc, List[ReidDesc], "ReidDescList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
