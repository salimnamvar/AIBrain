"""Object Re-identification Descriptor Module.
"""

# region Imported Dependencies
from datetime import datetime, timezone, timedelta
from typing import List, Union
import numpy as np
import numpy.typing as npt
from brain.util.obj import BaseObject, BaseObjectList

# region Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidDesc(BaseObject):
    def __init__(
        self, a_features: npt.NDArray[np.floating], a_extractor: str = "FeatExt", a_name: str = "ReidDesc"
    ) -> None:
        super().__init__(a_name=a_name)
        self._extractor: str = a_extractor
        self._features: np.ndarray = a_features
        self._timestamp: datetime = datetime.now().astimezone(tz=timezone(timedelta(hours=0)))

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def extractor(self) -> str:
        return self._extractor

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def to_dict(self) -> dict:
        dic = {"name": self.name, "extractor": self.extractor, "features": self.features, "timestamp": self.timestamp}
        return dic


# TODO(doc): Complete the document of following class
class ReidDescList(BaseObjectList[ReidDesc]):
    def __init__(
        self,
        a_name: str = "ReidDescList",
        a_max_size: int = -1,
        a_items: Union[ReidDesc, List[ReidDesc], "ReidDescList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
