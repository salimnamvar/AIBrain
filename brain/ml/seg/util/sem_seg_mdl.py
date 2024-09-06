"""Semantic Segmentation Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.cv.img import Image2D
from brain.cv.vid import Frame2D
from brain.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class SingSegModel(BaseModel, ABC):
    def __init__(self, a_name: str = "SingSegModel"):
        super().__init__(a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_frame: Frame2D, **kwargs) -> Image2D:
        NotImplementedError("Subclasses must implement `infer`")
