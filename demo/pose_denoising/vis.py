"""Visualization Module
"""

# region Imported Dependencies
import cv2
from brain.utils.cv.shape.ps import Pose2D
from brain.utils.cv.vid import Frame2D
from brain.utils.ml.seg import SegBBox2D
# endregion Imported Dependencies

def visualize(
    a_frame: Frame2D,
    a_conf_thre: float,
    a_box: SegBBox2D = None,
    a_pose: Pose2D = None,
    a_trk_pose: Pose2D = None,
):
    if a_box:
        a_box.p1.to_int()
        a_box.p2.to_int()
        cv2.rectangle(
            img=a_frame.data,
            pt1=a_box.p1.to_tuple(),
            pt2=a_box.p2.to_tuple(),
            color=(255, 180, 90),
            thickness=2,
            lineType=cv2.LINE_8,
        )

        if a_pose:
            lines = []
            for limb in a_pose.limbs:
                if limb.p1.score >= a_conf_thre and limb.p2.score >= a_conf_thre:
                    lines.append(limb.to_xy())
            cv2.polylines(a_frame.data, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

        if a_trk_pose:
            lines = []
            for limb in a_trk_pose.limbs:
                if limb.p1.score >= a_conf_thre and limb.p2.score >= a_conf_thre:
                    lines.append(limb.to_xy())
            cv2.polylines(a_frame.data, lines, False, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", a_frame.data)
    cv2.waitKey(1)