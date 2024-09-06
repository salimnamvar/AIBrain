"""Bounding Box Tracking Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import Optional

from brain.cv.shape.bx import BBox2DList
from brain.cv.vid import Frame2D
from brain.ml.reid.util import BaseReidModel
from brain.ml.trk.util import TrackedBBox2DList
from brain.ml.util import BaseModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class BBoxTrkModel(BaseModel, ABC):
    def __init__(self, a_reid: Optional[BaseReidModel] = None, a_name: str = "ObjTrkModel"):
        super().__init__(a_name)
        self.reid: BaseReidModel = a_reid

    @property
    def reid(self) -> BaseReidModel:
        """
        Getter for the re-identifier.

        Returns:
            PrsReid: The reid re-identifier.
        """
        return self._reid

    @reid.setter
    def reid(self, a_reid: BaseReidModel) -> None:
        """
        Setter for the re-identifier.

        Args:
            a_reid (BaseReidModel): The value to set for the re-identifier.

        Raises:
            TypeError: If `a_reid` is not of type PersonReid.
        """
        if a_reid is not None and not isinstance(a_reid, BaseReidModel):
            raise TypeError("The `a_reid` must be a `BaseReidModel`.")
        self._reid: BaseReidModel = a_reid

    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_boxes: BBox2DList, a_frame: Frame2D, **kwargs) -> TrackedBBox2DList:
        NotImplementedError("Subclasses must implement `infer`")
