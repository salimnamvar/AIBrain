"""Object Re-identification Utility Modules

This module provides subsequent utility modules of Object Re-identification models.
"""

# region Imported Dependencies
from brain.ml.reid.util.model.ov_reid_feat_ext_mdl import OVReidFeatExtModel
from .model import OVReidFeatExtModel, ReidFeatExtModel, BaseReidModel
from .assoc import MTE, MTEList, UMTList, UMEList, Associations
from .entity import (
    ReidDesc,
    ReidDescList,
    ReidTarget,
    ReidTargetList,
    ReidTargetNestedList,
    ReidTargetDict,
    ReidEntity,
    ReidEntityList,
    ReidEntityNestedList,
    ReidEntityDict,
    TypeReidEntity,
)

# endregion Imported Dependencies
