"""Bounding Box Tracking Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.util.cv.shape.bx import BBox2DList
from brain.util.cv.vid import Frame2D
from brain.util.ml.trk.util import TrackedBBox2DList
from brain.util.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class BBoxTrkModel(BaseModel, ABC):
    def __init__(self, a_name: str = "ObjTrkModel"):
        super().__init__(a_name)

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_boxes: BBox2DList, a_frame: Frame2D, **kwargs) -> TrackedBBox2DList:
        NotImplementedError("Subclasses must implement `infer`")
