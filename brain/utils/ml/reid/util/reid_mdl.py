"""Object Re-identification Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.utils.cv.img import Image2D
from . import ReidDesc
from ...util import BaseModel


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidModel(BaseModel, ABC):
    def __init__(self, a_name: str = "ObjReidModel"):
        super().__init__(a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs) -> ReidDesc:
        NotImplementedError("Subclasses must implement `infer`")
