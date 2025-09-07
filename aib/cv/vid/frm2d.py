"""2D Frame Module

    This Python file contains the definition of the Frame2D class, representing a 2D frame in the format of
    [Height, Width, Channels] derived from an :class:`Image2D` and typically part of a :class:`Video2D` sequence.
"""

# region Imported Dependencies
import uuid
from typing import Optional
import numpy as np
from aib.cv.img import Image2D
from aib.misc import Time


# endregion Imported Dependencies


class Frame2D(Image2D):
    """Frame2D

    This class defines a 2D frame in the format of [Height, Width, Channels] as an :class:`Image2D` that is from a
    :class:`Video2D`.

    Attributes
        id:
            A :class:`uuid.UUID` that specifies the ID of the Frame-2D object.
        video_id:
            A :class:`uuid.UUID` that specifies the ID of the :class:`Video2D` that the frame belongs to.
    """

    def __init__(
        self,
        a_data: np.ndarray,
        a_time: Time,
        a_filename: Optional[str] = None,
        a_id: Optional[uuid.UUID] = None,
        a_video_id: Optional[uuid.UUID] = None,
        a_name: str = "FRAME2D",
    ):
        """Constructor

        This constructor creates an instance of the Frame2D object.

        Args:
            a_data (np.ndarray):
                A NumPy array containing the frame data.
            a_time (Time):
                A Time object that specifies time information of the frame.
            a_filename (str, optional):
                A string specifying the filename of the frame (default is None).
            a_id (uuid.UUID, optional):
                A UUID specifying the ID of the Frame2D object (default is a randomly generated UUID).
            a_video_id (uuid.UUID, optional):
                A UUID specifying the ID of the Video2D to which the frame belongs (default is None).
            a_name (str, optional):
                A string specifying the name of the object (default is 'FRAME2D').

        Returns:
            None:
                The constructor does not return any values.

        Raises:
            TypeError:
                If the data types of `a_sequence_id`, `a_id`, or `a_video_id` are incorrect.
        """
        # region Input Checking
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError(f"`a_id` argument must be a `UUID` but it's type is `{type(a_id)}`")
        if a_video_id is not None and not isinstance(a_video_id, uuid.UUID):
            raise TypeError(f"`a_video_id` argument must be a `UUID` but it's type is `{type(a_video_id)}`")
        # endregion Input Checking

        super().__init__(a_data=a_data, a_filename=a_filename, a_name=a_name, a_time=a_time)
        self._id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        self._video_id: uuid.UUID = a_video_id

    @property
    def id(self) -> uuid:
        """Frame2D's ID Getter

        This property specifies the ID of the Frame2D object.

        Returns:
            uuid.UUID:
                A UUID specifying the ID of the Frame2D object.
        """
        return self._id

    @property
    def video_id(self) -> uuid.UUID:
        """Video2D's ID Getter

        This property specifies the ID of the :class:`Video2D` to which the frame belongs.

        Returns:
            uuid.UUID:
                A :class:`uuid.UUID` specifying the ID of the Video2D object.
        """
        return self._video_id
