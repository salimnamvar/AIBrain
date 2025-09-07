"""Population Target Container Modules
"""

# region Imported Dependencies
import uuid
from typing import List, Union

from aib.obj import BaseObjectDict
from .tgt import KFTargetList
from aib.ml.trk.OCSORT import PopulationDict as BasePopulationDict


# endregion Imported Dependencies


class PopulationDict(BasePopulationDict, BaseObjectDict[uuid.UUID, KFTargetList]):
    def __init__(
        self,
        a_name: str = "PopulationDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[KFTargetList, List[KFTargetList]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
