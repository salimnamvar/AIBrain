"""Kalman Filter Denoising Experiment

Using Pose Estimator to estimate the body skeleton and Kalman Filter to denoise the key-points.
"""

# region Imported Dependencies
import cv2
from brain.utils.cfg import BrainConfig
from brain.utils.cv.shape import Size
from brain.utils.cv.shape.ps import Pose2D
from brain.utils.cv.vid import Frame2D
from brain.utils.ml.pos.MNetSPTv4 import MNetSPTv4
from brain.utils.ml.seg import OVIS, SegBBox2DList
from examples.pose_denoising.trk import Tracker, State
from examples.pose_denoising.vis import visualize
# endregion Imported Dependencies


if __name__ == "__main__":
    cfg: BrainConfig = BrainConfig.get_instance(
        a_cfg="cfg.properties"
    )
    pose_estimator = MNetSPTv4(
        a_name="MNetSPTv4",
        a_mdl_path="saved_model.xml",
        a_mdl_device="CPU",
        a_conf_thre=0.0,
    )
    pose_estimator.load_mdl()
    person_detector = OVIS(
        a_name="ISP0007",
        a_mdl_path="instance-segmentation-person-0007.xml",
        a_mdl_device="CPU",
        a_conf_thre=0.1,
        a_nms_thre=0.45,
        a_top_k_thre=50,
        a_min_size_thre=Size(50, 50),
    )
    person_detector.load_mdl()
    video = cv2.VideoCapture("3.mp4")

    iter = 0
    trk: Tracker = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = Frame2D(frame)
        boxes: SegBBox2DList = person_detector.infer(a_image=frame)
        if len(boxes):
            pose: Pose2D = pose_estimator.infer(a_image=frame, a_box=boxes[0])
            if iter == 0:
                iter += 1
                trk = Tracker(
                    a_state=State(a_box=boxes[0], a_pose=pose),
                    a_conf_thre=0.0,
                    a_num_kps=17,
                )
            else:
                trk.update(a_state=State(boxes[0], pose))
        else:
            pose = None
        trk_pose = trk.predict()
        visualize(
            a_frame=frame,
            a_box=boxes[0],
            a_pose=pose,
            a_trk_pose=trk_pose,
            a_conf_thre=0,
        )
