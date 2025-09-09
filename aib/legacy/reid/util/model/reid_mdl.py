"""Object Re-identification Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import Generic
from aib.legacy.reid.util.entity import (
    ReidEntityDict,
    TypeReidEntityDict,
    TypeReidTarget,
    TypeReidTargetList,
    TypeReidTargetDict,
    TypeReidEntity,
)
from aib.legacy.reid.util.assoc import Associations
from aib.legacy.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class BaseReidModel(Generic[TypeReidEntityDict, TypeReidTargetDict, TypeReidEntity], BaseModel, ABC):
    def __init__(self, a_name: str = "ReidModel"):
        super().__init__(a_name)
        self.population: TypeReidEntityDict = ReidEntityDict()

    @abstractmethod
    def _cleanup(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `CLEANUP`")

    @abstractmethod
    def _associate(self, *args, a_tgt: TypeReidTargetList, **kwargs) -> Associations:
        NotImplementedError("Subclasses must implement `ASSOCIATE`")

    @abstractmethod
    def _assign(self, *args, a_tgt: TypeReidTargetList, a_assoc: Associations, **kwargs) -> TypeReidTargetDict:
        NotImplementedError("Subclasses must implement `ASSIGN`")

    @abstractmethod
    def infer(self, *args, a_tgt: TypeReidTargetList, **kwargs) -> TypeReidTargetDict:
        NotImplementedError("Subclasses must implement `INFER`")

    @abstractmethod
    def update(self, *args, a_ent: TypeReidTargetDict | TypeReidTargetList | TypeReidTarget, **kwargs) -> None:
        NotImplementedError("Subclasses must implement `UPDATE`")
