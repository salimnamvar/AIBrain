"""Reid0001 Object Re-identification Base Model
"""

from typing import Generic

# region Imported Dependencies
from brain.util.misc import TimeList
from .assign import ReidAssign
from .assoc import ReidAssoc
from .cleaner import ReidCleaner0002
from ..util import (
    BaseReidModel,
    Associations,
    ReidTargetList,
    ReidTargetDict,
    ReidTarget,
    ReidEntityDict,
    TypeReidEntity,
    ReidEntity,
)
from ..util.entity import TypeReidTargetDict, TypeReidTargetList, TypeReidTarget

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class Reid0001(BaseReidModel[ReidEntityDict, ReidTargetDict, TypeReidEntity]):
    def __init__(
        self,
        a_times: TimeList,
        a_cleanup_interval_mode: str = "step",
        a_cleanup_interval: int = 1920,
        a_auto_cleanup: bool = False,
        a_max_unmatched_age: int = 1920,
        a_num_desc: int = 100,
        a_num_state: int = -1,
        a_desc_samp_rate: int = 0,
        a_assoc_dist_metric: str = "correlation",
        a_assoc_dist_thre: float = 0.12,
        a_name: str = "ReidModel",
    ):
        super().__init__(a_name)
        self.population: ReidEntityDict = ReidEntityDict()
        self.auto_cleanup: bool = a_auto_cleanup
        self.association: ReidAssoc = ReidAssoc(a_dist_metric=a_assoc_dist_metric, a_dist_thre=a_assoc_dist_thre)
        self.assignment: ReidAssign[TypeReidEntity] = ReidAssign[TypeReidEntity](
            a_times=a_times,
            a_num_desc=a_num_desc,
            a_max_unmatched_age=a_max_unmatched_age,
            a_cleanup=not a_auto_cleanup,
            a_num_state=a_num_state,
            a_desc_samp_rate=a_desc_samp_rate,
            a_entity_type=ReidEntity,
        )
        if self.auto_cleanup:
            self.cleaner: ReidCleaner0002 = ReidCleaner0002(
                a_population=self.population,
                a_assignment=self.assignment,
                a_interval=a_cleanup_interval,
                a_interval_mode=a_cleanup_interval_mode,
                a_times=a_times,
            )

    def _cleanup(self) -> None:
        # CLEANUP
        if self.auto_cleanup:
            self.cleaner.infer()

    def _associate(self, *args, a_tgt: ReidTargetList, **kwargs) -> Associations:
        # ASSOCIATE
        associations: Associations = self.association.infer(a_tgt=a_tgt, a_ent=self.population)
        return associations

    def _assign(self, *args, a_tgt: ReidTargetList, a_assoc: Associations, **kwargs) -> TypeReidTargetDict:
        # ASSIGNMENT
        targets, _ = self.assignment.infer(a_tgt=a_tgt, a_ent=self.population, a_assoc=a_assoc)
        return targets

    def infer(self, *args, a_tgt: ReidTargetList, **kwargs) -> TypeReidTargetDict:
        # CLEANUP
        self._cleanup()
        # ASSOCIATE
        associations = self._associate(a_tgt=a_tgt)
        # ASSIGNMENT
        targets = self._assign(a_tgt=a_tgt, a_assoc=associations)
        return targets

    def update(self, *args, a_ent: TypeReidTargetDict | TypeReidTargetList | TypeReidTarget, **kwargs) -> None:
        # UPDATE
        if isinstance(a_ent, ReidTargetDict):
            a_ent = a_ent.values()
        elif isinstance(a_ent, ReidTarget):
            a_ent = [a_ent]

        for tgt in a_ent:
            self.population[tgt.id].update(a_inst=tgt)
