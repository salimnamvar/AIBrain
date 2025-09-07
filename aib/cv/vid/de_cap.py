"""Computer Vision - Video - Decord Video Reader

This module provides a wrapper around the Decord library's VideoReader functionality, allowing for efficient video frame
reading and manipulation.

Classes:
    DecordVideoReader:
        A class that encapsulates Decord's VideoReader functionality, providing methods to read frames,
        check properties, and release the video reader object.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu
from decord._ffi.ndarray import DECORDContext

from src.utils.cv.geom.size import Size
from src.utils.cv.img.frame import Frame2D
from src.utils.sys.b_obj import BaseObject


class DecordVideoReader(BaseObject):
    """Decord Video Reader

    This class provides an interface for reading video using the Decord library. It allows reading frames from a video
    file, handling various configurations such as frame size, number of threads, and fault tolerance. The class also
    supports properties to access video metadata like width, height, FPS, and total number of frames.

    Note: The `load` method should be called to initialize the video reader before reading frames.

    Attributes:
        src (str): The source of the video file.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
        size (Size[int]): The size of the video frame.
        shape (Tuple[int, int, int]): The shape of the video frame.
        fps (float): The frames per second of the video.
        num_frames (int): The total number of frames in the video.
        _ctx (DECORDContext): The context for Decord operations.
        _num_threads (int): Number of threads to use for video reading.
        _fault_tol (int): Fault tolerance for reading frames.
        _init_trial (int): Number of trials to initialize the video reader.
        _reader (VideoReader): The Decord video reader instance.
        _iter (int): Current iteration index for reading frames.
        _id (Optional[int]): Optional identifier for the video reader instance.
        _name (str): Name of the video reader instance.
    """

    def __init__(
        self,
        a_src: str,
        a_ctx: DECORDContext = cpu(0),
        a_size: Optional[Size[int]] = Size[int](-1, -1),
        a_num_threads: int = 0,
        a_fault_tol: int = -1,
        a_init_trial: int = 5,
        a_id: Optional[int] = None,
        a_name: str = 'DecordVideoReader',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Decord video reader.

        Args:
            a_src (str): The source of the video file.
            a_ctx (DECORDContext): The context for Decord operations.
            a_size (Size): The size of the video frame. If -1, it will be determined from the video.
            a_num_threads (int): Number of threads to use for video reading. If 0, it uses the default.
            a_fault_tol (int): Fault tolerance for reading frames. If -1, it will not retry on errors.
            a_init_trial (int): Number of trials to initialize the video reader.
            a_id (Optional[int]): Optional identifier for the video reader instance.
            a_name (str): Name of the video reader instance.
            a_use_prof (bool): Enable profiling for performance analysis.
            a_use_cfg (bool): Enable configuration loading.
            a_use_log (bool): Enable logging for debugging.
            **kwargs (Dict[str, Any]): Additional keyword arguments for BaseObject initialization.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )

        self._src: str = a_src
        self._ctx: DECORDContext = a_ctx
        self._size: Size[int] = a_size if a_size else Size[int](-1, -1)
        self._num_threads: int = a_num_threads
        self._fault_tol: int = a_fault_tol
        self._reader: VideoReader
        self._init_trial: int = a_init_trial
        self._iter: int = 0

    def load(self) -> None:
        """Load the Decord video reader.

        This method attempts to initialize the Decord video reader with the specified source and context. It retries
        for a specified number of trials if the initialization fails.

        Raises:
            RuntimeError: If the video source cannot be opened after the specified number of trials.
            Exception: If an error occurs during the initialization of the video reader.
        """
        for i in range(self._init_trial):
            try:
                self._reader = VideoReader(
                    uri=self._src,
                    ctx=self._ctx,
                    width=int(self._size.width if self._size.width > 0 else -1),
                    height=int(self._size.height if self._size.height > 0 else -1),
                    num_threads=self._num_threads,
                    fault_tol=self._fault_tol,
                )
                return
            except Exception as e:
                self.logger.warning("%s Initialization Trial: %d/%d: %s", self.name, i + 1, self._init_trial, e)
                time.sleep(5)
        raise RuntimeError(
            f"{self.name} Initialization Error: Failed to open video `{self._src}` after {self._init_trial} trials."
        )

    def read(self, a_index: Optional[int]) -> Tuple[bool, Frame2D | None]:
        """Read a frame from the video at the specified index.

        Args:
            a_index (Optional[int]): The index of the frame to read. If None, reads the next frame.

        Returns:
            Tuple[bool, Frame2D | None]:
                A tuple containing a success flag, the frame data, and the timestamp.
        """
        try:
            index = a_index if a_index is not None else self._iter

            if not self._reader or index >= len(self._reader):
                return False, None

            frame = self._reader[index].asnumpy()
            timestamp: float = float(np.average(self._reader.get_frame_timestamp(a_index)))
            self._iter = index + 1
            return True, Frame2D(
                data=frame, id=index, timestamp=timestamp, src_uri=self._src, src_id=self._id, src_name=self._name
            )

        except Exception as e:
            self.logger.error("%s Read Exception at frame %d: `%s`", self.name, a_index, e)
            return False, None

    def release(self) -> None:
        """Release the video reader.

        This method releases the resources held by the Decord video reader and resets the internal state."""
        del self._reader
        self._iter = 0
        self._reader = None

    def __reduce__(self):
        """Support for pickling the DecordVideoReader instance."""
        return (
            self.__class__,
            (
                self._src,
                self._ctx,
                self._size,
                self._num_threads,
                self._fault_tol,
                self._init_trial,
                self._id,
                self._name,
                self._use_prof,
                self._use_cfg,
                self._use_log,
            ),
        )

    @property
    def width(self) -> int:
        """Get the width of the video frame.

        Returns:
            int: The width of the video frame.
        """
        if self._size.width > 0:
            return int(self._size.width)
        return int(self._reader[0].shape[1]) if self._reader else 0

    @property
    def height(self) -> int:
        """Get the height of the video frame.

        Returns:
            int: The height of the video frame.
        """
        if self._size.height > 0:
            return int(self._size.height)
        return int(self._reader[0].shape[0]) if self._reader else 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the video frame.

        Returns:
            Tuple[int, int, int]: The shape of the video frame (height, width, channels).
        """
        return (self.height, self.width, 3)

    @property
    def size(self) -> Size[int]:
        """Get the size of the video frame.

        Returns:
            Size[int]: The size of the video frame with width and height as integers.
        """
        if self._size.width > 0 and self._size.height > 0:
            return self._size
        return Size[int](width=int(self.width), height=int(self.height))

    @property
    def fps(self) -> float:
        """Get the average frames per second of the video.

        Returns:
            float: The average frames per second of the video. Returns 0.0 if the reader is not initialized.
        """
        return float(self._reader.get_avg_fps()) if self._reader else 0.0

    @property
    def num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return len(self._reader) if self._reader else 0

    @property
    def src(self) -> str | int:
        """Get the video source.

        Returns:
            str | int: The source of the video, which can be a file path or an identifier.
        """
        return self._src

    def __len__(self) -> int:
        """Get the number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return self.num_frames
