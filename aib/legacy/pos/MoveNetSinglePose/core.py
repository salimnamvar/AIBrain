"""MoveNet-Single-Pose Pose Estimator

"""

# region Imported Dependencies
import cv2
import numpy as np

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.cv.shape.bx import BBox2D
from aib.cv.shape.ps import COCO17Pose2D
from aib.legacy.pos.util.ov_pos_mdl import OVSPosEstModel


# endregion Imported Dependencies

# TODO(doc): Complete the document of following class


class MoveNetSinglePose(OVSPosEstModel):
    def __init__(self, a_name: str, a_mdl_path: str, a_mdl_device: str, a_conf_thre: float):
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device, a_conf_thre=a_conf_thre)

    @property
    def mdl_inp_size(self) -> Size:
        """Getter for the input size of the model.

        Returns:
            Size: The input size of the model.

        Raises:
            TypeError: If the model is not loaded.
        """
        self.validate_mdl()
        return Size(a_height=self.mdl_inp_shape[1], a_width=self.mdl_inp_shape[2])

    def _preproc(self, a_image: Image2D, a_box: BBox2D) -> np.ndarray:
        image = a_image.data[
            max(0, a_box.p1.y) : min(a_image.height, a_box.p2.y),
            max(0, a_box.p1.x) : min(a_image.width, a_box.p2.x),
        ]

        image = cv2.resize(
            image,
            self.mdl_inp_size.to_tuple(),
            interpolation=cv2.INTER_AREA,
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # add batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def _postproc(self, a_preds: np.ndarray, a_box: BBox2D) -> COCO17Pose2D:
        # Extract estimations from the output layer
        kps = np.squeeze(a_preds)  # (17x3) -> [y, x, score]
        kps = kps[:, [1, 0, 2]]  # [x, y, score]

        # Denormalize
        kps[:, 0:2] *= (a_box.width, a_box.height)
        # Transformation
        kps[:, 0:2] += (a_box.p1.x, a_box.p1.y)

        # Create Pose2D
        pose = COCO17Pose2D.from_xys(a_coordinates=kps)

        return pose

    def infer(self, a_image: Image2D, a_box: BBox2D) -> COCO17Pose2D:
        # Pre-process input image
        proc_input = self._preproc(a_image=a_image, a_box=a_box)

        # Model inference
        preds = self.mdl.infer_new_request(proc_input)[0]

        # Post-process predictions
        pose = self._postproc(a_preds=preds, a_box=a_box)

        return pose
