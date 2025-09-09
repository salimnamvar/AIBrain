"""Reid0001 Object Re-identification Association Model
"""

# region Imported Dependencies
from typing import Tuple
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from aib.legacy.reid.util.entity.ent import ReidEntityDict
from aib.legacy.reid.util import ReidDescList, ReidDesc, Associations, MTE, ReidTargetList
from aib.legacy.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidAssoc(BaseModel):
    def __init__(
        self, a_dist_metric: str = "correlation", a_dist_thre: float = 0.12, a_name: str = "ReidAssoc"
    ) -> None:
        super().__init__(a_name=a_name)
        self.dist_metric: str = a_dist_metric
        self.dist_thre: float = a_dist_thre

    # region DISTANCE CALCULATION
    def _cal_dist(self, a_tgt_desc: ReidDesc, a_ent_desc: ReidDesc) -> float:
        dist = +np.inf
        if a_tgt_desc.extractor == a_ent_desc.extractor:
            dist = cdist(a_tgt_desc.features.reshape(1, -1), a_ent_desc.features.reshape(1, -1), self.dist_metric)
        return dist

    def _cal_min_dist(self, a_tgt_desc: ReidDescList, a_ent_desc: ReidDescList) -> float:
        min_dist: float = +np.inf
        for tgt_desc in a_tgt_desc:
            for ent_desc in a_ent_desc:
                dist = self._cal_dist(a_tgt_desc=tgt_desc, a_ent_desc=ent_desc)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _cal_dist_matrix(self, a_tgt: ReidTargetList, a_ent: ReidEntityDict) -> Tuple[npt.NDArray[np.floating], dict]:
        dist_matrix = np.zeros((len(a_tgt), len(a_ent)), dtype=np.float32)
        ent_id_map = {}
        for i, tgt in enumerate(a_tgt):
            for j, (ent_id, ent) in enumerate(a_ent.items()):
                if i == 0:
                    ent_id_map[j] = ent_id
                dist = self._cal_min_dist(a_tgt_desc=tgt.descriptors, a_ent_desc=ent.descriptors)
                dist_matrix[i, j] = dist
        return dist_matrix, ent_id_map

    # endregion DISTANCE CALCULATION

    def _associate(self, a_tgt: ReidTargetList, a_ent: ReidEntityDict) -> Associations:
        # PRE-PROCESS
        assoc = Associations()

        if len(a_tgt):
            # DISTANCE MATRIX
            dist_matrix, ent_id_map = self._cal_dist_matrix(a_tgt=a_tgt, a_ent=a_ent)

            # MATCHED TARGET-ENTITY PAIRS
            tgt_inds, ent_inds = linear_sum_assignment(dist_matrix)
            to_del = []
            for index, (i, j) in enumerate(zip(tgt_inds, ent_inds)):
                if dist_matrix[i, j] < self.dist_thre:
                    assoc.matched_pairs.append(MTE(a_tgt=i, a_ent=ent_id_map[j]))
                else:
                    to_del.append(index)

            # Remove the uncertain matched pairs
            if to_del:
                tgt_inds = np.delete(tgt_inds, to_del)
                ent_inds = np.delete(ent_inds, to_del)

            # UNMATCHED TARGETS
            unmatched_tgt_inds = set(range(len(a_tgt))) - set(tgt_inds)
            for i in unmatched_tgt_inds:
                # Append if the target has extracted features
                if len(a_tgt[i].descriptors):
                    assoc.unmatched_targets.append(i)

            # UNMATCHED ENTITIES
            unmatched_ent_inds = set(range(len(a_ent))) - set(ent_inds)
            for j in unmatched_ent_inds:
                assoc.unmatched_entities.append(ent_id_map[j])
        return assoc

    def infer(self, a_tgt: ReidTargetList, a_ent: ReidEntityDict) -> Associations:
        # ASSOCIATE
        assoc = self._associate(a_tgt=a_tgt, a_ent=a_ent)
        return assoc
