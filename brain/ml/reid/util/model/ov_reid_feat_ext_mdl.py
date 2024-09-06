"""OpenVino Object Re-identification Feature Extractor Model
"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from brain.cv.img import Image2D
from brain.ml.util import OVModel

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class OVReidFeatExtModel(OVModel, ABC):
    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
    ):
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device)

    def to_dict(self) -> dict:
        dic = {
            "name": self.name,
            "mdl_path": self.mdl_path,
            "mdl_device": self.mdl_device,
        }
        return dic

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, a_image: Image2D, **kwargs):
        NotImplementedError("Subclasses must implement `infer`")
