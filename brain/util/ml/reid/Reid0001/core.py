"""Reid0001 Object Re-identification Base Model
"""

# region Imported Dependencies
import threading
import time
from .assign import ReidAssign
from .assoc import ReidAssoc
from ..util import BaseReidModel, Associations, ReidTargetList, ReidTargetDict

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class Reid0001(BaseReidModel):
    def __init__(
        self,
        a_cleanup_interval: int = 3600,
        a_auto_cleanup: bool = True,
        a_max_unmatched_age: int = 3600,
        a_num_desc: int = 100,
        a_assoc_dist_metric: str = "correlation",
        a_assoc_dist_thre: float = 0.12,
        a_name: str = "ReidModel",
    ):
        super().__init__(a_name)
        self.cleanup_interval: int = a_cleanup_interval
        self.auto_cleanup: bool = a_auto_cleanup
        self.max_unmatched_age: int = a_max_unmatched_age
        self.association: ReidAssoc = ReidAssoc(a_dist_metric=a_assoc_dist_metric, a_dist_thre=a_assoc_dist_thre)
        self.assignment: ReidAssign = ReidAssign(
            a_num_desc=a_num_desc, a_max_unmatched_age=a_max_unmatched_age, a_online_cleanup=not a_auto_cleanup
        )
        self._init_cleanup()

    def _init_cleanup(self):
        # AUTO-CLEANUP INITIALIZATION
        if self.auto_cleanup:
            self._cleanup_lock = threading.Lock()
            self._cleanup_condition = threading.Condition(lock=self._cleanup_lock)
            self._cleanup_in_progress: bool = False
            self._cleanup_thread = threading.Thread(target=self._cleanup, daemon=True)
            self._cleanup_thread.start()

    def _cleanup(self) -> None:
        # AUTO-CLEANUP
        while self.auto_cleanup:
            with self._cleanup_lock:
                self._cleanup_in_progress = True
                self.assignment.delete(a_ent=self.population, a_unmatched_entities=None)
                self._cleanup_condition.notify_all()
                self._cleanup_in_progress = False
            time.sleep(self.cleanup_interval)

    def associate(self, *args, a_tgt: ReidTargetList, **kwargs) -> Associations:
        # ASSOCIATE
        associations: Associations = self.association.infer(a_tgt=a_tgt, a_ent=self.population)
        return associations

    def assign(self, *args, a_tgt: ReidTargetList, a_assoc: Associations, **kwargs) -> ReidTargetDict:
        # ASSIGNMENT
        targets, self.population = self.assignment.infer(a_tgt=a_tgt, a_ent=self.population, a_assoc=a_assoc)
        return targets

    def infer(self, *args, a_tgt: ReidTargetList, **kwargs) -> ReidTargetDict:
        with self._cleanup_condition:
            # CLEANUP
            while self._cleanup_in_progress:
                self._cleanup_condition.wait()

            # ASSOCIATE
            associations = self.associate(a_tgt=a_tgt)
            # ASSIGNMENT
            targets = self.assign(a_tgt=a_tgt, a_assoc=associations)
        return targets
