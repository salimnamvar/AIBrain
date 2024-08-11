"""Object Re-identification Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.util.ml.reid.util.entity.tgt import ReidTargetDict, ReidTargetList
from brain.util.ml.reid.util.entity import ReidEntityDict
from brain.util.ml.reid.util.assoc import Associations
from brain.util.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class BaseReidModel(BaseModel, ABC):
    def __init__(self, a_name: str = "ReidModel"):
        super().__init__(a_name)
        self.population: ReidEntityDict = ReidEntityDict()

    @abstractmethod
    def associate(self, *args, a_tgt: ReidTargetList, **kwargs) -> Associations:
        NotImplementedError("Subclasses must implement `ASSOCIATE`")

    @abstractmethod
    def assign(self, *args, a_tgt: ReidTargetList, a_assoc: Associations, **kwargs) -> ReidTargetDict:
        NotImplementedError("Subclasses must implement `ASSIGN`")

    @abstractmethod
    def infer(self, *args, a_tgt: ReidTargetList, **kwargs) -> ReidTargetDict:
        NotImplementedError("Subclasses must implement `INFER`")
