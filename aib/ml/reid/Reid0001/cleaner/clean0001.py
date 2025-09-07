"""Reid0001 Object Re-identification Memory Cleaner

"""

# region Imported Dependencies
import threading
import time
from aib.misc import TimeList
from aib.ml.reid.Reid0001.assign import ReidAssign
from aib.ml.reid.Reid0001.cleaner.clean_mdl import ReidCleanerModel
from aib.ml.reid.util import ReidEntityDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidCleaner0001(ReidCleanerModel):
    def __init__(
        self,
        a_times: TimeList,
        a_population: ReidEntityDict,
        a_assignment: ReidAssign,
        a_interval: int = 1920,
        a_interval_mode: str = "time",
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
        self._cleanup_lock = threading.Lock()
        self._cleanup_condition = threading.Condition(lock=self._cleanup_lock)
        self._cleanup_in_progress: bool = False
        self._cleanup_thread = threading.Thread(target=self._cleanup, daemon=True)
        self._stop_thread: bool = False
        self.start()

    def _wait(self):
        if self.interval_mode == "time":
            for _ in range(self.interval):
                if self._stop_thread:
                    break
                time.sleep(1)
        elif self.interval_mode == "step":
            while True:
                interval_statuses = [(c_time.step % self.interval) == 0 for c_time in self.times]
                if any(interval_statuses) or self._stop_thread:
                    break
                time.sleep(1)

    def _cleanup(self) -> None:
        while not self._stop_thread:
            with self._cleanup_lock:
                self._cleanup_in_progress = True
                self.assignment.delete(a_ent=self.population, a_unmatched_entities=None)
                self._cleanup_condition.notify_all()
                self._cleanup_in_progress = False
            self._wait()

    def start(self) -> None:
        self._cleanup_thread.start()

    def stop(self) -> None:
        self._stop_thread = True
        with self._cleanup_condition:
            self._cleanup_condition.notify_all()
        self._cleanup_thread.join()

    def infer(self) -> None:
        with self._cleanup_condition:
            while self._cleanup_in_progress:
                self._cleanup_condition.wait()
