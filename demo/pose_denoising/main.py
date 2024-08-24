"""Pose Denoising

Using Pose Estimator to estimate the body skeleton and Kalman Filter to denoise the key-points.
"""

# region Imported Dependencies
import cv2
from brain.util.cv.shape import Size
from brain.util.cv.shape.ps import Pose2D
from brain.util.cv.vid import Frame2D
from brain.util.ml.pos.MNetSPTv4 import MNetSingPose
from brain.util.ml.seg import OVIS, SegBBox2DList
from demo.pose_denoising.trk import Tracker, State
from demo.pose_denoising.vis import visualize
# endregion Imported Dependencies


if __name__ == "__main__":
    pose_estimator = MNetSingPose(
        a_name="MNetSPTv4",
        a_mdl_path=r"G:\Models\movenet\singlepose-lightning\4\4.xml",
        a_mdl_device="CPU",
        a_conf_thre=0.0,
    )
    pose_estimator.load_mdl()
    person_detector = OVIS(
        a_name="ISP0007",
        a_mdl_path=r"G:\Models\intel\instance-segmentation-person-0007\FP32\instance-segmentation-person-0007.xml",
        a_mdl_device="CPU",
        a_conf_thre=0.1,
        a_nms_thre=0.45,
        a_top_k_thre=50,
        a_min_size_thre=Size(50, 50),
    )
    person_detector.load_mdl()
    video = cv2.VideoCapture(r"G:\Research\AIBrain\Data\1.mp4")

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
                trk_pose = trk.state()
            else:
                trk.update(a_state=State(boxes[0], pose))
                trk_pose = trk.state()
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
