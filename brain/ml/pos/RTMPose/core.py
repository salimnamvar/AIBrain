"""RTMPose Pose Estimator

"""

# region Imported Dependencies
from typing import Tuple, Union, List

import cv2
import numpy as np
import numpy.typing as npt
from openvino.runtime.utils.data_helpers import OVDict

from brain.cv.img import Image2D
from brain.cv.shape import Size
from brain.cv.shape.bx import BBox2D
from brain.cv.shape.ps import COCO17Pose2D
from brain.cv.shape.pt import Point2D
from brain.ml.pos.RTMPose.util import wrap_affine, extract_keypoints
from brain.ml.pos.util.ov_pos_mdl import OVSPosEstModel


# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class RTMPose(OVSPosEstModel):
    def __init__(
        self,
        a_name: str,
        a_mdl_path: str,
        a_mdl_device: str,
        a_conf_thre: float,
        a_norm_mean: Union[Tuple[float, float, float], List[float], npt.NDArray[np.float32]] = (
            123.675,
            116.28,
            103.53,
        ),
        a_norm_std: Union[Tuple[float, float, float], List[float], npt.NDArray[np.float32]] = (58.395, 57.12, 57.375),
        a_padding_ratio: float = 1.25,
        a_simcc_split_ratio: float = 2.0,
    ):
        super().__init__(a_name=a_name, a_mdl_path=a_mdl_path, a_mdl_device=a_mdl_device, a_conf_thre=a_conf_thre)
        self.norm_mean: npt.NDArray[np.float32] = np.array(a_norm_mean)
        self.norm_std: npt.NDArray[np.float32] = np.array(a_norm_std)
        self.padding_ratio: float = a_padding_ratio
        self.simcc_split_ratio: float = a_simcc_split_ratio

    @property
    def padding_ratio(self) -> float:
        return self._padding_ratio

    @padding_ratio.setter
    def padding_ratio(self, a_padding_ratio: float):
        if a_padding_ratio is None or not isinstance(a_padding_ratio, float):
            raise TypeError("The `a_padding_ratio` should be a `float`.")
        self._padding_ratio: float = a_padding_ratio

    @property
    def simcc_split_ratio(self) -> float:
        return self._simcc_split_ratio

    @simcc_split_ratio.setter
    def simcc_split_ratio(self, a_simcc_split_ratio: float):
        if a_simcc_split_ratio is None or not isinstance(a_simcc_split_ratio, float):
            raise TypeError("The `a_simcc_split_ratio` should be a `float`.")
        self._simcc_split_ratio: float = a_simcc_split_ratio

    @property
    def norm_mean(self) -> npt.NDArray[np.float32]:
        return self._norm_mean

    @norm_mean.setter
    def norm_mean(self, a_norm_mean: Union[Tuple[float, float, float], List[float], npt.NDArray[np.float32]]):
        if a_norm_mean is None:
            raise ValueError("norm_mean cannot be None.")
        if isinstance(a_norm_mean, np.ndarray):
            arr = a_norm_mean
        else:
            arr = np.asarray(a_norm_mean, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("norm_mean must have a length of 3.")
        self._norm_mean = arr

    @property
    def norm_std(self) -> npt.NDArray[np.float32]:
        return self._norm_std

    @norm_std.setter
    def norm_std(self, a_norm_std: Union[Tuple[float, float, float], List[float], npt.NDArray[np.float32]]):
        if a_norm_std is None:
            raise ValueError("norm_std cannot be None.")
        if isinstance(a_norm_std, np.ndarray):
            arr = a_norm_std
        else:
            arr = np.asarray(a_norm_std, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("norm_std must have a length of 3.")
        self._norm_std = arr

    def _preproc(self, a_image: Image2D, a_box: BBox2D) -> Tuple[npt.NDArray[np.floating], Point2D, Size]:
        # Pad
        center: Point2D = a_box.center
        size: Size = a_box.size * self.padding_ratio

        # Top-Down Affine Transform
        image, size = wrap_affine(a_src_size=size, a_src_center=center, a_dst_size=self.mdl_inp_size, a_image=a_image)

        # Normalize
        image = ((image - self.norm_mean) / self.norm_std).astype(np.float32)

        # Color Transfer
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image to Tensor
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = image[None, :, :, :]
        return image, center, size

    def _postproc(self, a_preds: OVDict, a_box_center: Point2D, a_box_size: Size) -> COCO17Pose2D:
        # Decode SimCC Feature Maps
        simcc_x, simcc_y = a_preds[self.mdl.outputs[0]], a_preds[self.mdl.outputs[1]]
        kps, scores = extract_keypoints(simcc_x, simcc_y)
        kps = kps / self.simcc_split_ratio

        # Rescale keypoints
        kps = kps / self.mdl_inp_size.to_numpy() * a_box_size.to_numpy()
        kps = kps + a_box_center.to_numpy() - a_box_size.to_numpy() / 2

        # Combine Keypoints and Scores
        coords = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)
        coords = coords.reshape((17, 3))

        # Create Pose2D
        pose = COCO17Pose2D.from_xys(a_coordinates=coords)

        return pose

    def infer(self, a_image: Image2D, a_box: BBox2D) -> COCO17Pose2D:
        # Pre-process input image
        input, box_center, box_size = self._preproc(a_image=a_image, a_box=a_box)

        # Model inference
        output = self.mdl(input)

        # Post-process predictions
        pose = self._postproc(a_preds=output, a_box_center=box_center, a_box_size=box_size)

        return pose
