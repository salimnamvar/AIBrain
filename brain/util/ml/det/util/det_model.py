"""Object Detection Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import List

from brain.util.cv.img import Image2D
from brain.util.cv.shape.bx import BBox2DList
from brain.util.ml.util import BaseModel, BaseModelList
from brain.util.obj import BaseObjectList


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ObjDetModel(BaseModel, ABC):
    def __init__(self, a_name: str = "ObjDetModel"):
        super().__init__(a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        NotImplementedError("Subclasses must implement `infer`")


# TODO(doc): Complete the document of following class
class ObjDetModelList(BaseModelList, BaseObjectList[ObjDetModel], ABC):
    def __init__(
        self,
        a_name: str = "ObjDetModelList",
        a_max_size: int = -1,
        a_items: List[ObjDetModel] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> BBox2DList:
        NotImplementedError("Subclasses must implement `infer`")
