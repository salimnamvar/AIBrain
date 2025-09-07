"""Computer Vision - Video - OpenCV Video Capture

This module provides a wrapper around OpenCV's VideoCapture functionality,
allowing for easy video frame capture, preprocessing, and iteration. It
supports synchronous and asynchronous modes, different I/O modes, and
optional profiling, configuration, and logging integration.

Classes:
    OpenCVVideoCapture: Encapsulates OpenCV VideoCapture with enhanced interface.

Type Variables:
    IOT: Type variable for input/output operations.
"""

import asyncio
import time
import warnings
from typing import Any, Iterator, Literal, Optional, Tuple, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from src.utils.cnt.io import BaseIO
from src.utils.cv.geom.size import IntSize, Size
from src.utils.cv.img.frame import Frame2D
from src.utils.misc.common_types import StopEvent
from src.utils.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Frame2D | None, Any])


class OpenCVVideoCapture(BaseModel[IOT]):
    """OpenCV Video Capture Wrapper

    This class wraps OpenCV's VideoCapture functionality, providing a flexible and extensible
    interface for handling video streams. It supports different execution, concurrency, I/O, and
    processing modes, as well as optional profiling, configuration, and logging integrations.

    Attributes:
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        size (Size): Size of the video frames if explicitly set, otherwise derived from the capture.
        shape (Tuple[int, int, int]): Shape of the video frames (height, width, channels).
        is_opened (bool): Whether the video capture is currently opened.
        fps (float): Frames per second of the video source.
        num_frames (int): Total number of frames in the video (if available).
        src (str | int): Video source, either a file path or a camera index.
        init_trial (int): Number of trials to attempt when opening the video source.
        id (int): Unique identifier for the video capture instance.
        name (str): Name of the video capture instance.
        _cap (cv2.VideoCapture): Underlying OpenCV VideoCapture object.
        _cap_api (int): OpenCV backend API for video capture (e.g., cv2.CAP_ANY).
    """

    def __init__(
        self,
        a_src: str | int,
        a_cap_api: int = cv2.CAP_ANY,
        a_size: Optional[IntSize] = None,
        a_init_trial: int = 5,
        a_target_fps: Optional[int] = None,
        a_src_mode: Literal["file", "camera"] = "file",
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["sys", "opencv"] = "opencv",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = 1,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'OpenCVVideoCapture',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the video capture object.

        Args:
            a_src (str | int): Video source (file path or camera index).
            a_cap_api (int): OpenCV capture API backend (default: cv2.CAP_ANY).
            a_size (Optional[Size[int]]): Target frame size for resizing.
            a_init_trial (int): Number of attempts to open the source.
            a_target_fps (Optional[int]): Target FPS for capture.
            a_src_mode (Literal["file", "camera"]): Source type.
            a_call_mode (Literal["sync", "async"]): Execution mode.
            a_io_mode (Literal["args", "queue", "ipc"]): Input/output mode.
            a_proc_mode (Literal["batch", "online"]): Processing mode.
            a_backend (Literal["sys", "opencv"]): Processing backend.
            a_conc_mode (Optional[Literal["thread", "process"]]): Concurrency mode.
            a_max_workers (Optional[int]): Maximum workers for concurrency.
            a_io (Optional[IOT]): Input/output interface.
            a_stop_event (Optional[StopEvent]): Stop event for async execution.
            a_id (Optional[int]): Unique identifier for the instance.
            a_name (str): Name of the instance.
            a_use_prof (bool): Enable profiling.
            a_use_cfg (bool): Enable configuration.
            a_use_log (bool): Enable logging.
            **kwargs: Additional arguments passed to BaseModel.
        """
        super().__init__(
            a_call_mode=a_call_mode,
            a_io_mode=a_io_mode,
            a_proc_mode=a_proc_mode,
            a_backend=a_backend,
            a_conc_mode=a_conc_mode,
            a_max_workers=a_max_workers,
            a_io=a_io,
            a_stop_event=a_stop_event,
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._src: str | int = a_src
        self._size: Optional[IntSize] = a_size
        self._cap_api: int = a_cap_api
        self._cap: Optional[cv2.VideoCapture] = None
        self._init_trial: int = a_init_trial
        self._target_fps: Optional[int] = a_target_fps
        self._src_mode: Literal["file", "camera"] = a_src_mode

    def load(self) -> None:
        """Open the video source and initialize the capture object.

        Raises:
            RuntimeError: If the video source cannot be opened after
                the specified number of trials.
        """
        try:
            for _ in range(self._init_trial):
                self._cap = cv2.VideoCapture(self._src, self._cap_api)
                if not self._cap.isOpened():
                    msg = f"{self.name} Initialization Error: Unable to open video source `{self._src}`"
                    warnings.warn(msg)
                    time.sleep(5)
                else:
                    break
            if not self._cap.isOpened():
                msg = f"{self.name} Initialization Error: Unable to open video source `{self._src}`"
                raise RuntimeError(msg)

            if self._target_fps is not None and self._target_fps > 0:
                self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        except Exception as e:
            msg = f"{self.name} Initialization Exception: `{e}`"
            raise RuntimeError(msg) from e

    def _preproc(self, a_frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Preprocess Frame

        This method preprocesses the frame by resizing it to the specified size if provided.

        Args:
            a_frame: The frame to preprocess.

        Returns:
            npt.NDArray[np.uint8]: The preprocessed frame.
        """
        if self._size is not None:
            a_frame = cv2.resize(a_frame, self._size.to_tuple(), interpolation=cv2.INTER_LINEAR)
        return a_frame

    def read(self) -> Tuple[bool, Frame2D | None]:
        """Read a frame from the video capture.

        Returns:
            Tuple[bool, Frame2D | None]:
                A tuple containing a boolean indicating success and the frame data. If the read fails, the frame data
                will be None.

        Raises:
            RuntimeError: If the video capture is not opened or has been released.
            Exception: If an error occurs while reading the frame.
        """
        try:
            if not self._cap or not self._cap.isOpened():
                msg = f"{self.name} Read Error: Video capture is not opened or has been released."
                warnings.warn(msg)
                return False, None

            ret, frame = self._cap.read()
            frame = self._preproc(frame)
            frame_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = Frame2D(
                data=frame,
                id=frame_idx,
                timestamp=timestamp_sec,
                src_uri=self._src,
                src_id=self._id,
                src_name=self._name,
            )

            return ret, frame
        except Exception as e:
            msg = f"{self.name} Read Exception: `{e}`"
            warnings.warn(msg)
            return False, None

    def __iter__(self) -> Iterator[Tuple[bool, Frame2D | None]]:
        """Get an iterator over the frames in the video capture.

        Returns:
            Iterator[Tuple[bool, Frame2D | None]]: Iterator over the frames.
        """
        if not self._cap or not self._cap.isOpened():
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        frame_duration = 0.0
        tick_freq = 0.0
        if self._target_fps is not None:
            frame_duration = 1 / self._target_fps if self._target_fps and self._target_fps > 0 else 0
            tick_freq = cv2.getTickFrequency()
        for _ in range(self.num_frames):
            start_tick = cv2.getTickCount()
            ret, frame = self.read()
            if self._target_fps is not None:
                end_tick = cv2.getTickCount()
                elapsed_time = (end_tick - start_tick) / tick_freq
                remaining_time = frame_duration - elapsed_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
            yield (ret, frame)

    def __len__(self) -> int:
        """Get the number of frames in the video capture.

        Returns:
            int: Number of frames in the video capture.
        """
        if not self._cap or not self._cap.isOpened():
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        return self.num_frames

    def release(self) -> None:
        """Release the video capture object."""
        self._cap.release()

    def __reduce__(self):
        return (self.__class__, (self._src, self._cap_api, self._id, self._init_trial, self._name))

    @property
    def width(self) -> int:
        """Get the width of the video frames.

        Returns:
            int: Width of the video frames.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get the height of the video frames.

        Returns:
            int: Height of the video frames.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Size[int]:
        """Get the size of the video frames.

        Returns:
            Size[int]: Size of the video frames with width and height as integers.
        """
        return Size[int](width=int(self.width), height=int(self.height))

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the video frames.

        Returns:
            Tuple[int, int, int]: Shape of the video frames (height, width, channels).
        """
        return (self.height, self.width, 3)

    @property
    def is_opened(self) -> bool:
        """Check if the video capture is opened.

        Returns:
            bool: True if the video capture is opened, False otherwise.
        """
        return self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get the frames per second of the video capture.

        Returns:
            float: Frames per second of the video capture.
        """
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def num_frames(self) -> int:
        """Get the total number of frames in the video capture.

        Returns:
            int: Total number of frames in the video capture.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def src(self) -> str | int:
        """Get the video source.
        Returns:
            str | int: The video source (file path or camera index)."""
        return self._src

    async def infer_async(self) -> None:
        """Perform asynchronous frame reading and input streaming.

        This method reads frames from the video source asynchronously and sends
        them to the configured I/O interface. It supports frame rate regulation
        if `self._target_fps` is set. The method will stop if `self.stop_event`
        is set or when the end of a file-based video is reached.

        Preconditions:
            - `self.stop_event` must be set.
            - `self.io` must be initialized.
            - `self.call_mode` must be 'async'.
            - `self._io_mode` must be 'queue'.

        Raises:
            AssertionError: If `stop_event` or `io` are not set.
            NotImplementedError: If the configuration is not supported.
            RuntimeError: If an exception occurs during asynchronous reading
                or streaming, the stop event is triggered, and the exception
                is re-raised.
        """
        assert self.stop_event is not None, "Stop event must be set."
        assert self.io is not None, "IO must be set."

        try:
            if self.call_mode == 'async' and self._io_mode == "queue":
                frame_duration = 0.0
                tick_freq = 0.0
                if self._target_fps is not None:
                    frame_duration = 1 / self._target_fps if self._target_fps and self._target_fps > 0 else 0
                    tick_freq = cv2.getTickFrequency()
                try:
                    while not self.stop_event.is_set():
                        start_tick = cv2.getTickCount()
                        ret, frame = await asyncio.to_thread(self.read)
                        if not ret:
                            if self._src_mode == "file":
                                break
                            if self._src_mode == "camera":
                                continue
                        await self.io.put_input_async(a_input=frame)

                        if self._target_fps is not None:
                            end_tick = cv2.getTickCount()
                            elapsed_time = (end_tick - start_tick) / tick_freq
                            remaining_time = frame_duration - elapsed_time
                            if remaining_time > 0:
                                await asyncio.sleep(remaining_time)
                finally:
                    await self.io.input_done_async()
            else:
                raise NotImplementedError("Configuration not supported.")
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Async Inference failed: {e}") from e

    async def run_async(self) -> None:
        """Run asynchronous video capture and frame streaming with multiple workers.

        This method launches multiple asynchronous tasks to read frames from the video
        source and stream them to the configured I/O interface. It supports controlling
        the number of concurrent workers via `self.max_workers` and ensures that all
        input streams are properly marked as done after completion.

        Preconditions:
            - `self.stop_event` must be set.
            - `self.io` must be initialized.
            - `self.call_mode` must be 'async'.
            - `self.io_mode` must be 'queue'.
            - `self.max_workers` must be set to a positive integer.

        Raises:
            AssertionError: If `stop_event`, `io`, or `max_workers` are not properly set.
            NotImplementedError: If the configuration is not supported.
            RuntimeError: If an exception occurs during asynchronous execution, the
                stop event is triggered, and the exception is re-raised.

        Notes:
            Each worker executes `infer_async` concurrently to process video frames.
            After all workers finish or if an error occurs, the input queue is marked
            as done via `input_done_async`.
        """
        assert self.stop_event is not None, "Stop event must be set."

        try:
            if self.call_mode == 'async' and self.io_mode == "queue":
                assert (
                    self.max_workers is not None and self.max_workers > 0
                ), "Max workers must be set and greater than 0."
                assert self.io is not None, "IO must be set."

                try:
                    async with asyncio.TaskGroup() as tg:
                        for wid in range(self.max_workers):
                            tg.create_task(self.infer_async(), name=f"worker-{wid}")
                finally:
                    await self.io.input_done_async()
            else:
                raise NotImplementedError("Configuration not supported.")
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Async Run failed: {e}") from e
