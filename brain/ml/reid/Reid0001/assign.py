"""Reid0001 Object Re-identification Assignment Model
"""

# region Imported Dependencies
import uuid
from typing import Tuple, Generic, Type
from brain.misc import TimeList
from brain.ml.reid.util import (
    ReidTargetDict,
    ReidTargetList,
    Associations,
    ReidEntityDict,
    UMEList,
    MTEList,
    ReidDescList,
    UMTList,
    TypeReidEntity,
    ReidEntity,
)
from brain.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidAssign(Generic[TypeReidEntity], BaseModel):
    def __init__(
        self,
        a_times: TimeList,
        a_num_desc: int,
        a_num_state: int = -1,
        a_desc_samp_rate: int = 0,
        a_max_unmatched_age: int = 1920,
        a_cleanup: bool = True,
        a_entity_type: Type[TypeReidEntity] = ReidEntity,
        a_name: str = "ReidAssign",
    ):
        super().__init__(a_name=a_name)
        self.desc_samp_rate: int = a_desc_samp_rate
        self.times: TimeList = a_times
        self.cleanup: bool = a_cleanup
        self.num_desc: int = a_num_desc
        self.num_state: int = a_num_state
        self.max_unmatched_age: int = a_max_unmatched_age
        self.entity_type: Type[TypeReidEntity] = a_entity_type

    def insert(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_unmatched_targets: UMTList
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        targets = ReidTargetDict()
        for tgt_ind in a_unmatched_targets:
            if a_tgt[tgt_ind].descriptors is not None and len(a_tgt[tgt_ind].descriptors):
                # INSERT entity
                time = a_tgt[tgt_ind].time.copy()
                ent = self.entity_type(
                    a_time=time,
                    a_state=a_tgt[tgt_ind].state,
                    a_num_state=self.num_state,
                    a_id=uuid.uuid4(),
                    a_descriptors=ReidDescList(a_max_size=self.num_desc, a_items=a_tgt[tgt_ind].descriptors),
                    a_num_desc=self.num_desc,
                    a_desc_samp_rate=self.desc_samp_rate,
                )
                a_ent.append(a_key=ent.id, a_value=ent)

                # UPDATE target
                a_tgt[tgt_ind].update(a_inst=ent)
                targets.append(a_key=ent.id, a_value=a_tgt[tgt_ind])
        return targets, a_ent

    def update(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_matched_pairs: MTEList
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        targets = ReidTargetDict()
        for pair in a_matched_pairs:
            # UPDATE entity
            a_ent[pair.ent].update(a_inst=a_tgt[pair.tgt])

            # UPDATE target
            a_tgt[pair.tgt].update(a_inst=a_ent[pair.ent])
            targets.append(a_key=a_tgt[pair.tgt].id, a_value=a_tgt[pair.tgt])
        return targets, a_ent

    def delete(self, a_ent: ReidEntityDict, a_unmatched_entities: UMEList = None) -> ReidEntityDict:
        if a_unmatched_entities is None:
            a_unmatched_entities = UMEList(a_items=list(a_ent.keys()))
        for ent_ind in a_unmatched_entities:
            # DELETE entity
            unmatched_statuses = [
                (c_time.step - a_ent[ent_ind].states.last_state.time.step) > self.max_unmatched_age
                for c_time in self.times
            ]
            if all(unmatched_statuses):
                a_ent.pop(a_key=ent_ind)
        return a_ent

    def assign(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_assoc: Associations
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        targets = ReidTargetDict()
        if a_tgt:
            # INSERT UNMATCHED TARGETS
            inserted_targets, a_ent = self.insert(
                a_tgt=a_tgt, a_ent=a_ent, a_unmatched_targets=a_assoc.unmatched_targets
            )
            targets.update(a_dict=inserted_targets)

            # UPDATE MATCHED PAIRS
            updated_targets, a_ent = self.update(a_tgt=a_tgt, a_ent=a_ent, a_matched_pairs=a_assoc.matched_pairs)
            targets.update(a_dict=updated_targets)

            # DELETE UNMATCHED ENTITIES
            if self.cleanup:
                a_ent = self.delete(a_ent=a_ent, a_unmatched_entities=a_assoc.unmatched_entities)
        return targets, a_ent

    def infer(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_assoc: Associations
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        # ASSIGNMENT
        targets, a_ent = self.assign(a_tgt=a_tgt, a_ent=a_ent, a_assoc=a_assoc)
        return targets, a_ent
