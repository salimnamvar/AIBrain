"""Video

    This Python file contains the definition of the :class:`Video2D` class for capturing video from files,
    image sequences, or cameras.
"""

# region Imported Dependencies
import uuid
from typing import Union, Tuple

import cv2

from brain.utils.cv.shape.sz import Size
from brain.utils.cv.vid import Frame2D
from brain.utils.obj import BaseObject


# endregion Imported Dependencies


class Video2D(BaseObject):
    """Video2D

    This Python file contains the definition of the Video2D class, which represents a 2D video object for capturing
     video from files, image sequences, or cameras.

    Attributes:
        name (str, optional):
            A string specifying the name of the Video2D object (default is 'VIDEO2D').
        source (Union[str, int]):
            A string or integer value indicating the resource file or camera number.
        number_of_frames (int):
            An integer representing the total number of frames in the video.
        current_frame_id (int):
            An integer representing the current frame's sequential ID in the reading process (starts from 0).
        id (uuid.UUID):
            A UUID specifying the ID of the Video2D object.
        backend (int):
            An integer specifying the video capture backend (e.g., :class:`cv2.CAP_FFMPEG`, :class:`cv2.CAP_V4L2`).
        inc_fr_num (bool):
            A bool indicates whether the frame counter is set to be incremented.
    """

    def __init__(
        self,
        a_source: Union[str, int],
        a_id: uuid.UUID = None,
        a_backend=cv2.CAP_FFMPEG,
        a_name="VIDEO2D",
        a_inc_fr_num: bool = False,
    ):
        """Video2D Constructor

        Initializes a Video2D object for capturing video from files, image sequences, or cameras.

        Args:
            a_source (Union[str, int]):
                A string or integer value indicating the resource file or camera number.
            a_id (uuid.UUID, optional):
                A UUID specifying the ID of the Video2D object (default is generated if not provided).
            a_backend (int, optional):
                An integer specifying the video capture backend (e.g., cv2.CAP_FFMPEG) (default is cv2.CAP_FFMPEG).
            a_name (str, optional):
                A string specifying the name of the object (default is 'VIDEO2D').
            a_inc_fr_num (bool, optional):
                Indicates whether the frame counter should be incremented. Defaults to False.

        Returns:
            None

        Raises:
            TypeError:
                If the data type of `a_id` is incorrect.

        Notes:
            This constructor initializes the Video2D object, including its source, ID, backend, and name.
            It also initializes the video capture process by calling the `initialize_video` method.
        """

        # region Input Checking
        if a_id is not None and not isinstance(a_id, uuid.UUID):
            raise TypeError(
                f"`a_id` argument must be a `UUID` but it's type is `{type(a_id)}`"
            )
        # endregion Input Checking

        super().__init__(a_name)

        self._id: uuid.UUID = a_id if a_id is not None else uuid.uuid4()
        self._current_frame_id: int = -1
        self.source: Union[str, int] = a_source
        self.backend = a_backend
        self.inc_fr_num: bool = a_inc_fr_num
        self.video_capture: cv2.VideoCapture = None
        self.initialize_video()

    def initialize_video(self):
        """Initialize Video Capture

        Initializes the video capture process using OpenCV based on the provided source and backend.

        Raises an exception if an error occurs during initialization.

        Raises:
            Exception:
                If an unexpected error occurs during video initialization.

        Notes:
            This method is responsible for initializing the video capture process using the specified source and
            backend. It handles data type conversion for the source, and any unexpected OpenCV errors are
            captured and raised as exceptions.
        """
        try:
            if isinstance(self.source, str):
                if self.source.isdigit():
                    self.source = int(self.source)
            self.video_capture = cv2.VideoCapture(self.source, self.backend)
        except cv2.error as e:
            raise Exception(
                f"An Unexpected OpenCV error occurred during initializing the video. The error is `{e}`"
            )
        except Exception as e:
            raise Exception(
                f"An Unexpected OpenCV error occurred during initializing the camera. The error is `{e}`"
            )

    def read(self) -> Tuple[bool, Frame2D]:
        """Read Video Frame

        Reads a single frame from the video capture and returns it as a Frame2D object.

        Returns:
            Tuple[bool, :class:`Frame2D`]:
                A tuple containing a boolean value (True if a frame was successfully read) and the captured frame
                as a :class:`Frame2D` object.

        Raises:
            Exception:
                If an unexpected error occurs during frame reading.

        Notes:
            This method reads a single frame from the video capture using OpenCV. It returns a tuple where the first
            element is a boolean indicating whether the frame was successfully read, and the second element is the
            captured frame represented as a :class:`Frame2D` object. In case of any unexpected OpenCV errors during
            frame reading, an exception is raised.
        """
        try:
            ret, frame = self.video_capture.read()
        except cv2.error as e:
            raise Exception(
                f"An Unexpected OpenCV error occurred during reading video's frame. The error is `{e}`"
            )
        except Exception as e:
            raise Exception(
                f"An Unexpected OpenCV error occurred during reading video's frame. The error is `{e}`"
            )

        if ret:
            frame_seq_id = None
            if self.inc_fr_num:
                self._current_frame_id += 1
                frame_seq_id = self._current_frame_id
            frame = Frame2D(
                a_data=frame, a_video_id=self._id, a_sequence_id=frame_seq_id
            )
        return ret, frame

    @property
    def width(self) -> int:
        """Video Frame Width Getter

        Property that returns the width of the video frames.

        Returns:
            int:
                An integer representing the width of the video frames.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Video Frame Height Getter

        Property that returns the height of the video frames.

        Returns:
            int:
                An integer representing the height of the video frames.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Size:
        """Video's Size Getter

        This property specifies the size of the frames in [Width, Height] format.

        Returns:
            Size:
                The size of the image as :class:`Size`.
        """
        return Size(self.width, self.height, a_name=f"{self.name} Size")

    @property
    def is_opened(self) -> bool:
        """Video Is Opened Getter

        Property that returns whether the video capture is opened and ready for reading frames.

        Returns:
            bool:
                True if the video capture is opened and ready, otherwise False.
        """
        return self.video_capture.isOpened()

    @property
    def fps(self) -> int:
        """Video Frames Per Second (FPS) Getter

        Property that returns the frames per second (FPS) of the video.

        Returns:
            int:
                An integer representing the frames per second of the video.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))

    def __reduce__(self):
        """Serialization Method

        Special method for object serialization, used to save and load Video2D objects.

        Returns:
            Tuple:
                A tuple representing the class constructor and its arguments for serialization.
        """
        return self.__class__, (self.source, self.id, self.backend, self.name)

    @property
    def number_of_frames(self) -> int:
        """Number of Frames Getter

        Property that returns the total number of frames in the video.

        Returns:
            int:
                An integer representing the total number of frames in the video.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def current_frame_id(self) -> int:
        """Current Frame's ID Getter

        Property that returns the sequential ID of the current frame during video frame reading.

        Returns:
            int:
                An integer representing the sequential ID of the current frame being read from the video.
        """
        return self._current_frame_id

    @property
    def id(self) -> uuid:
        """ID Getter

        Property that returns the ID of the Video2D object.

        Returns
            :class:`uuid.UUID`:
                A :class:`uuid.UUID` representing the ID of the Video2D object.
        """
        return self._id

    def to_dict(self) -> dict:
        """Convert to Dictionary

        Method that represents the Video2D object as a dictionary.

        Returns:
            dict:
                A dictionary containing the Video2D object's name and its associated :class:`uuid.UUID`.
        """
        dic = {self.name: self._id}
        return dic
