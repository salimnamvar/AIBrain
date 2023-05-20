"""2D Video

This python file contains the 2D Video class object.
"""


# region Imported Dependencies
import uuid
from copy import deepcopy
import cv2
from brain.inp.vid.frm2d import Frame2D
# endregion Imported Dependencies


class Video2D(cv2.VideoCapture):
    def __init__(self, *args, a_id: uuid.UUID = None, **kwargs):
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError("`a_id` argument must be a `UUID`")
        self._id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        self.__current_frame_id: int = 0
        super().__init__(*args, **kwargs)

    def read(self):
        ret, frame = super().read()
        if ret:
            self.__current_frame_id += 1
            frame = Frame2D(frame, a_sequence_id=self.__current_frame_id, a_video_id=self._id)
        return ret, frame

    @property
    def id(self) -> uuid:
        return self._id

    def copy(self) -> 'Video2D':
        return deepcopy(self)
