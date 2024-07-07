"""Segmented Instance Tracked Module
"""

# region Import Dependencies
import uuid
from datetime import datetime
from typing import Optional, List, Union

from brain.utils.cv.img import Image2D
from brain.utils.cv.shape import Size
from brain.utils.cv.shape.pt import Point2D
from brain.utils.ml.seg import SegBBox2D, SegBBox2DList
from brain.utils.obj import BaseObjectList, BaseObjectDict
from .bbox import TrackedBBox2D, TrackedBBox2DList, TrackedBBox2DDict


# endregion Import Dependencies


# TODO(doc): Complete the document of following class
class TrackedSegBBox2D(TrackedBBox2D, SegBBox2D):
    def __init__(
        self,
        a_id: uuid.UUID,
        a_timestamp: Optional[datetime],
        a_p1: Point2D,
        a_p2: Point2D,
        a_score: float,
        a_mask: Image2D,
        a_label: int,
        a_img_size: Optional[Size] = None,
        a_strict: Optional[bool] = False,
        a_conf_thre: Optional[float] = None,
        a_min_size_thre: Optional[Size] = None,
        a_do_validate: Optional[bool] = True,
        a_name: str = "TrackedSegBBox2D",
    ):
        SegBBox2D.__init__(
            self,
            a_p1=a_p1,
            a_p2=a_p2,
            a_score=a_score,
            a_mask=a_mask,
            a_label=a_label,
            a_img_size=a_img_size,
            a_strict=a_strict,
            a_conf_thre=a_conf_thre,
            a_min_size_thre=a_min_size_thre,
            a_do_validate=a_do_validate,
            a_name=a_name,
        )
        self.id: uuid.UUID = a_id
        self.timestamp: datetime = a_timestamp


# TODO(doc): Complete the document of following class
class TrackedSegBBox2DList(
    SegBBox2DList, TrackedBBox2DList, BaseObjectList[TrackedSegBBox2D]
):
    def __init__(
        self,
        a_name: str = "TrackedSegBBox2DList",
        a_max_size: int = -1,
        a_items: List[TrackedSegBBox2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class TrackedSegBBox2DDict(
    TrackedBBox2DDict, BaseObjectDict[uuid.UUID, TrackedSegBBox2D]
):
    def __init__(
        self,
        a_name: str = "TrackedSegBBox2DDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[TrackedSegBBox2D, List[TrackedSegBBox2D]] = None,
    ):
        super().__init__(a_name, a_max_size, a_key, a_value)
