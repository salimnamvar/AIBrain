"""Memory Cleaner Base Model

"""

# region Imported Dependencies
from abc import ABC, abstractmethod
from aib.misc import TimeList
from aib.ml.reid.Reid0001.assign import ReidAssign
from aib.ml.reid.util import ReidEntityDict
from aib.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidCleanerModel(BaseModel, ABC):
    def __init__(
        self,
        a_times: TimeList,
        a_population: ReidEntityDict,
        a_assignment: ReidAssign,
        a_interval: int = 1920,
        a_interval_mode: str = "step",
        a_name: str = "ReidCleaner",
    ) -> None:
        super().__init__(a_name=a_name)
        self.times: TimeList = a_times
        self.population: ReidEntityDict = a_population
        self.assignment: ReidAssign = a_assignment
        self.interval_mode: str = a_interval_mode
        self.interval: int = a_interval

    @property
    def interval_mode(self) -> str:
        return self._interval_mode

    @interval_mode.setter
    def interval_mode(self, a_interval_mode: str) -> None:
        if a_interval_mode is None or a_interval_mode not in ["time", "step"]:
            raise TypeError("The `a_interval_mode` must be a `str` and one of [`time`, `step`] options.")
        self._interval_mode: str = a_interval_mode

    @abstractmethod
    def _cleanup(self) -> None:
        NotImplementedError("Subclasses must implement `CLEANUP`")

    @abstractmethod
    def infer(self) -> None:
        NotImplementedError("Subclasses must implement `INFERENCE`")
