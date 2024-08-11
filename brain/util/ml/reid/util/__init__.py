"""Object Re-identification Utility Modules

This module provides subsequent utility modules of Object Re-identification models.
"""

# region Imported Dependencies
from brain.util.ml.reid.util.entity.desc import ReidDesc, ReidDescList
from brain.util.ml.reid.util.model.ov_reid_feat_ext_mdl import OVReidFeatExtModel
from .entity import (
    ReidDesc,
    ReidDescList,
    ReidEntityState,
    ReidEntityStateList,
    ReidEntityStateDict,
    ReidTarget,
    ReidTargetList,
    ReidTargetNestedList,
    ReidTargetDict,
    ReidEntity,
    ReidEntityList,
    ReidEntityNestedList,
    ReidEntityDict,
)
from .model import OVReidFeatExtModel, ReidFeatExtModel, BaseReidModel
from .assoc import MTE, MTEList, UMTList, UMEList, Associations

# endregion Imported Dependencies
