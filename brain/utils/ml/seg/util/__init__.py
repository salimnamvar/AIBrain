"""Object Segmentation Utility Modules

This module provides subsequent utility modules of object segmentation models.
"""

# region Imported Dependencies
from .bbox import SegBBox2D, SegBBox2DList, SegBBox2DNestedList
from .ov_inst_seg_mdl import OVInstSegModel, OVInstSegModelList
from .sem_seg_mdl import SingSegModel
from .inst_seg_mdl import InstSegModel

# endregion Imported Dependencies
