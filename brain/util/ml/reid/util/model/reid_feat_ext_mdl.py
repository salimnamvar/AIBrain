"""Object Re-identification Feature Extractor Base Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.util.cv.img import Image2D
from brain.util.ml.util import BaseModel


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class ReidFeatExtModel(BaseModel, ABC):
    def __init__(self, a_name: str = "ReidFeatExtModel"):
        super().__init__(a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs):
        NotImplementedError("Subclasses must implement `infer`")
