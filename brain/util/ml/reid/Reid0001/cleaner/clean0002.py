"""Clean0002 Memory Cleaner

"""

# region Imported Dependencies
from brain.util.misc import TimeList
from brain.util.ml.reid.Reid0001.assign import ReidAssign
from brain.util.ml.reid.Reid0001.cleaner.clean_mdl import ReidCleanerModel
from brain.util.ml.reid.util import ReidEntityDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidCleaner0002(ReidCleanerModel):
    def __init__(
        self,
        a_times: TimeList,
        a_population: ReidEntityDict,
        a_assignment: ReidAssign,
        a_interval: int = 1920,
        a_interval_mode: str = "step",
        a_name: str = "ReidCleaner",
    ) -> None:
        super().__init__(
            a_name=a_name,
            a_times=a_times,
            a_population=a_population,
            a_assignment=a_assignment,
            a_interval_mode=a_interval_mode,
            a_interval=a_interval,
        )
        self.last_times: TimeList = a_times

    def _cleanup(self) -> None:
        if self.interval_mode == "step":
            interval_statuses = [
                (curr_time.step - last_time.step) >= self.interval
                for curr_time, last_time in zip(self.times, self.last_times)
            ]
        elif self.interval_mode == "time":
            interval_statuses = [
                (curr_time.timestamp - last_time.timestamp).total_seconds() >= self.interval
                for curr_time, last_time in zip(self.times, self.last_times)
            ]
        else:
            raise TypeError("The `a_interval_mode` must be a `str` and one of [`time`, `step`] options.")

        if all(interval_statuses):
            self.assignment.delete(a_ent=self.population, a_unmatched_entities=None)
            self.last_times = self.times.copy()

    def infer(self):
        self._cleanup()
