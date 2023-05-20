"""2D Frame

This python file contains the 2D Frame class object.
"""


# region Imported Dependencies
import uuid
from brain.inp.img.img2d import Image2D
# endregion Imported Dependencies


class Frame2D(Image2D):
    def __init__(self, *args, a_sequence_id: int = None, a_id: uuid.UUID = None, a_video_id: uuid.UUID = None,
                 **kwargs):
        if a_sequence_id is not None and not isinstance(a_sequence_id, int):
            raise TypeError("`a_sequence_id` argument must be an `int`")
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError("`a_id` argument must be a `UUID`")
        if a_video_id is not None and not isinstance(a_video_id, uuid.UUID):
            raise TypeError("`a_video_id` argument must be a `UUID`")
        self._sequence_id: int = a_sequence_id
        self._id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        self._video_id: uuid.UUID = a_video_id

    def __new__(cls, *args, a_sequence_id: int = None, a_id: uuid.UUID = None, a_video_id: uuid.UUID = None, **kwargs):
        if a_sequence_id is not None and not isinstance(a_sequence_id, int):
            raise TypeError("`a_sequence_id` argument must be an `int`")
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError("`a_id` argument must be a `UUID`")
        if a_video_id is not None and not isinstance(a_video_id, uuid.UUID):
            raise TypeError("`a_video_id` argument must be a `UUID`")
        cls._sequence_id: int = a_sequence_id
        cls._id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        cls._video_id: uuid.UUID = a_video_id
        obj = super().__new__(cls, *args, **kwargs)
        return obj

    @property
    def id(self) -> uuid:
        return self._id

    @property
    def sequence_id(self) -> int:
        return self._sequence_id

    @property
    def video_id(self) -> uuid.UUID:
        return self._video_id
