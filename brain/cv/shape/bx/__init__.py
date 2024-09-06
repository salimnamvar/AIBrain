""" Box Modules

This module contains sub-utility modules for handling operations related to boxes.
"""

# region Imported Dependencies
from .bbox import BBox2D, BBox2DList, BBox2DNestedList
from .box import Box2D, Box2DList
from .fmt import xywh_to_xyxy, xyxy_to_xywh, xyxy_to_cxywh, cxywh_to_xyxy, cxyar_to_xyxy
from .status import CoordStatus, ConfidenceStatus, SizeStatus

# region Imported Dependencies
