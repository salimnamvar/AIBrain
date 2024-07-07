"""Instance Segmentation Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.utils.cv.img import Image2D
from .bbox import SegBBox2DList
from ...det import ObjDetModel


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class InstSegModel(ObjDetModel, ABC):
    def __init__(
        self,
        a_name: str,
    ) -> None:
        super().__init__(a_name=a_name)

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> SegBBox2DList:
        NotImplementedError("Subclasses must implement `infer`")
