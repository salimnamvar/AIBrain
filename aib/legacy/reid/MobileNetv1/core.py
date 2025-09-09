"""MobileNet-v1 Object Re-identification Feature Extractor Model
desc: cut model from layer MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6
"""

# region Imported Dependencies
import cv2
import numpy as np
from openvino.runtime.utils.data_helpers import OVDict
from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.legacy.reid.util import OVReidFeatExtModel, ReidDesc

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class MobileNetv1(OVReidFeatExtModel):
    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
    ) -> None:
        super().__init__(
            a_name=a_name,
            a_mdl_path=a_mdl_path,
            a_mdl_device=a_mdl_device,
        )

    @property
    def mdl_inp_size(self) -> Size:
        self.validate_mdl()
        return Size(a_height=self.mdl_inp_shape[1], a_width=self.mdl_inp_shape[2])

    def _preproc(self, a_image: Image2D) -> np.ndarray:
        self.validate_mdl()

        # Resize Image
        image = cv2.resize(
            a_image.data,
            self.mdl_inp_size.to_tuple(),
            interpolation=cv2.INTER_AREA,
        )

        # Add batch dimension
        image = np.expand_dims(image, 0)

        return image

    def _postproc(self, a_preds: OVDict, a_image: Image2D) -> ReidDesc:
        self.validate_mdl()
        desc = ReidDesc(
            a_features=a_preds["MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0"].flatten(),
            a_extractor=self.name,
            a_time=a_image.time.copy(),
        )
        return desc

    def infer(self, *args, a_image: Image2D, **kwargs) -> ReidDesc:
        self.validate_mdl()

        # Pre-process input image
        proc_input = self._preproc(a_image=a_image)

        # Model inference
        preds = self.mdl.infer_new_request(proc_input)

        # Post-process predictions
        desc = self._postproc(a_preds=preds, a_image=a_image)

        return desc
