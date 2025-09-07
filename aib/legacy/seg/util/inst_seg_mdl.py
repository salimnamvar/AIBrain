"""Instance Segmentation Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import List

from aib.cv.img import Image2D
from aib.obj import BaseObjectList
from .bbox import SegBBox2DList
from ...det import ObjDetModel, ObjDetModelList


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class InstSegModel(ObjDetModel, ABC):
    def __init__(
        self,
        a_name: str = "InstSegModel",
    ) -> None:
        super().__init__(a_name=a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        NotImplementedError("Subclasses must implement `infer`")


# TODO(doc): Complete the document of following class
class InstSegModelList(ObjDetModelList, BaseObjectList[InstSegModel], ABC):
    def __init__(
        self,
        a_name: str = "InstSegModelList",
        a_max_size: int = -1,
        a_items: List[ObjDetModel] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        NotImplementedError("Subclasses must implement `infer`")
