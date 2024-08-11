"""Reid0001 Object Re-identification Assignment Model
"""

# region Imported Dependencies
import uuid
from datetime import datetime, timezone, timedelta
from typing import Tuple

from brain.util.ml.reid.util import (
    ReidTargetDict,
    ReidTargetList,
    Associations,
    ReidEntityDict,
    UMEList,
    ReidEntityState,
    MTEList,
    ReidEntity,
    ReidDescList,
    UMTList,
)
from brain.util.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidAssign(BaseModel):
    def __init__(
        self, a_num_desc: int, a_max_unmatched_age: int, a_online_cleanup: bool = True, a_name: str = "ReidAssign"
    ):
        super().__init__(a_name=a_name)
        self.online_cleanup: bool = a_online_cleanup
        self.num_desc: int = a_num_desc
        self.max_unmatched_age: int = a_max_unmatched_age

    def insert(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_unmatched_targets: UMTList
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        targets = ReidTargetDict()
        for tgt_ind in a_unmatched_targets:
            if a_tgt[tgt_ind].descriptors is not None and len(a_tgt[tgt_ind].descriptors):
                # INSERT entity
                ent = ReidEntity(
                    a_state=ReidEntityState(),
                    a_id=uuid.uuid4(),
                    a_descriptors=ReidDescList(a_max_size=self.num_desc, a_items=a_tgt[tgt_ind].descriptors),
                )
                a_ent.append(a_key=ent.id, a_value=ent)

                # UPDATE target
                a_tgt[tgt_ind].update(a_timestamp=ent.timestamp, a_id=ent.id)
                targets.append(a_key=ent.id, a_value=a_tgt[tgt_ind])
        return targets, a_ent

    def update(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_matched_pairs: MTEList
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        targets = ReidTargetDict()
        for pair in a_matched_pairs:
            # UPDATE entity
            a_ent[pair.ent].update(a_state=ReidEntityState(), a_descriptors=a_tgt[pair.tgt].descriptors)

            # UPDATE target
            a_tgt[pair.tgt].update(a_timestamp=a_ent[pair.ent].timestamp, a_id=a_ent[pair.ent].id)
            targets.append(a_key=a_tgt[pair.tgt].id, a_value=a_tgt[pair.tgt])
        return targets, a_ent

    def delete(self, a_ent: ReidEntityDict, a_unmatched_entities: UMEList = None) -> ReidEntityDict:
        curr_time: datetime = datetime.now().astimezone(tz=timezone(timedelta(hours=0)))
        if a_unmatched_entities is None:
            a_unmatched_entities = UMEList(a_items=list(a_ent.keys()))
        for ent_ind in a_unmatched_entities:
            # DELETE entity
            unmatched_age = curr_time - a_ent[ent_ind].last_state.timestamp
            if unmatched_age.total_seconds() >= self.max_unmatched_age:
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
            if self.online_cleanup:
                a_ent = self.delete(a_ent=a_ent, a_unmatched_entities=a_assoc.unmatched_entities)
        return targets, a_ent

    def infer(
        self, a_tgt: ReidTargetList, a_ent: ReidEntityDict, a_assoc: Associations
    ) -> Tuple[ReidTargetDict, ReidEntityDict]:
        # ASSIGNMENT
        targets, a_ent = self.assign(a_tgt=a_tgt, a_ent=a_ent, a_assoc=a_assoc)
        return targets, a_ent
