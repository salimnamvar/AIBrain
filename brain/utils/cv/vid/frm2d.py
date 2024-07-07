"""2D Frame Module

    This Python file contains the definition of the Frame2D class, representing a 2D frame in the format of
    [Height, Width, Channels] derived from an :class:`Image2D` and typically part of a :class:`Video2D` sequence.
"""

# region Imported Dependencies
import uuid

import numpy as np

from brain.utils.cv.img import Image2D


# endregion Imported Dependencies


class Frame2D(Image2D):
    """Frame2D

    This class defines a 2D frame in the format of [Height, Width, Channels] as an :class:`Image2D` that is from a
    :class:`Video2D`.

    Attributes
        id:
            A :class:`uuid.UUID` that specifies the ID of the Frame-2D object.
        sequence_id:
            A :type:`uuid.UUID` that specifies the sequence based ID of the frame that shows the index of the
            frame in a sequence such as an instance of :class:`Video2D`.
        video_id:
            A :class:`uuid.UUID` that specifies the ID of the :class:`Video2D` that the frame belongs to.
    """

    def __init__(
        self,
        a_data: np.ndarray,
        a_filename: str = None,
        a_sequence_id: int = None,
        a_id: uuid.UUID = None,
        a_video_id: uuid.UUID = None,
        a_name: str = "FRAME2D",
    ):
        """Constructor

        This constructor creates an instance of the Frame2D object.

        Args:
            a_data (np.ndarray):
                A NumPy array containing the frame data.
            a_filename (str, optional):
                A string specifying the filename of the frame (default is None).
            a_sequence_id (int, optional):
                An integer value indicating the sequential ID of the frame within a Video2D (default is None).
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
        if a_sequence_id is not None and not isinstance(a_sequence_id, int):
            raise TypeError(
                f"`a_sequence_id` argument must be an `int` but it's type is `{type(a_sequence_id)}`"
            )
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError(
                f"`a_id` argument must be a `UUID` but it's type is `{type(a_id)}`"
            )
        if a_video_id is not None and not isinstance(a_video_id, uuid.UUID):
            raise TypeError(
                f"`a_video_id` argument must be a `UUID` but it's type is `{type(a_video_id)}`"
            )
        # endregion Input Checking

        super().__init__(a_data=a_data, a_filename=a_filename, a_name=a_name)
        self._sequence_id: int = a_sequence_id
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
    def sequence_id(self) -> int:
        """Frame2D's Sequential ID Getter

        This property specifies the sequence ID of the Frame2D object.

        Returns:
            int:
                An integer indicating the sequence ID of the Frame2D object.
        """
        return self._sequence_id

    @property
    def video_id(self) -> uuid.UUID:
        """Video2D's ID Getter

        This property specifies the ID of the :class:`Video2D` to which the frame belongs.

        Returns:
            uuid.UUID:
                A :class:`uuid.UUID` specifying the ID of the Video2D object.
        """
        return self._video_id
