"""Object Detection Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC

from brain.utils.cv.img import Image2D
from brain.utils.cv.shape.bx import BBox2DList
from brain.utils.ml.util import BaseModel

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
