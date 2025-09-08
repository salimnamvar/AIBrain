"""Dataset - Video Dataset Loader Utilities

This module provides a unified interface for loading video files using either Decord or OpenCV as the backend.

Classes:
    VideoDatasetLoader:
        A class that encapsulates the functionality of loading video files and managing video readers.

Type Variables:
    IOT: Type variable for input/output operations.
"""

from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence, Tuple, TypeVar

import cv2

from aib.cnt.b_dict import BaseDict
from aib.cnt.io import BaseIO
from aib.cv.geom.size import IntSize
from aib.cv.img.frame import Frame2D
from aib.cv.vid.cv_cap import OpenCVVideoCapture
from aib.misc.common_types import StopEvent
from aib.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Frame2D | None, Any])


class VideoDatasetLoader(BaseModel[IOT]):
    """Video Dataset Loader

    This class provides a unified interface for loading video files using either Decord or OpenCV as the backend.
    It scans a specified argsory for video files, initializes the appropriate video reader for each file,
    and stores them in a dictionary for easy access.

    Attributes:
        _size (Size[int]): Size of the video frames.
        _file_ext (Sequence[str]): Video file extensions to look for.
        _dataset_dir (PathLike[str]): argsory containing video files.
        _backend (str): Backend to use for video reading ('decord' or 'opencv').
        _data_load ers (BaseDict[str, Any]): Dictionary to store video reader instances.
    """

    def __init__(
        self,
        a_dataset_dir: str | PathLike[str],
        a_file_ext: Sequence[Literal["mp4", "avi", "mov", "mkv", "wmv"]] = ("mp4",),
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
        a_name: str = 'VideoDatasetLoader',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the video data loader.

        Args:
            a_dataset_dir (str | PathLike[str]): argsory containing video files.
            a_size (Optional[Size[int]]): Size of the video frames. If None, frames will not be resized.
            a_ext (Sequence[Literal["mp4", "avi", "mov", "mkv", "wmv"]]): Video file extensions to look for.
            a_backend (Literal["decord", "opencv"]): Backend to use for video reading.
            a_id (Optional[int]): Optional identifier for the data loader instance.
            a_name (str): Name of the data loader instance.
            a_use_prof (bool): Enable profiling.
            a_use_cfg (bool): Enable configuration loading.
            a_use_log (bool): Enable logging.
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
        self._size: Optional[IntSize] = a_size
        self._file_ext: Sequence[str] = a_file_ext
        self._cap_api: int = a_cap_api
        self._init_trial: int = a_init_trial
        self._target_fps: Optional[int] = a_target_fps
        self._src_mode: Literal["file", "camera"] = a_src_mode
        self._dataset_dir: Path = Path(a_dataset_dir).resolve()
        self._backend = a_backend
        self._loaders: Optional[BaseDict[str, OpenCVVideoCapture]] = None

    def load(self) -> None:
        """Load the dataset by creating video reader instances for each video file.

        This method scans the dataset argsory for video files with the specified extension
        and initializes a video reader for each file (using either DecordVideoReader or OpenCVVideoCapture),
        storing them in the `_data_loaders` dictionary.

        Note: Each data loader in the dictionary should be loaded before use.
        """
        self._loaders = BaseDict[str, OpenCVVideoCapture]()

        for i, path in enumerate(
            sorted(
                chain.from_iterable(
                    self._dataset_dir.rglob(f"*.{ext}")
                    for ext in {ext.lower() for ext in self._file_ext} | {ext.upper() for ext in self._file_ext}
                )
            )
        ):
            rel_path = str(path.relative_to(self._dataset_dir))
            if self._backend == "opencv":
                video_reader = OpenCVVideoCapture(
                    a_src=str(path),
                    a_cap_api=self._cap_api,
                    a_size=self._size,
                    a_init_trial=self._init_trial,
                    a_target_fps=self._target_fps,
                    a_src_mode=self._src_mode,
                    a_call_mode=self._call_mode,
                    a_io_mode=self._io_mode,
                    a_proc_mode=self._proc_mode,
                    a_backend=self._backend,
                    a_conc_mode=self._conc_mode,
                    a_max_workers=self._max_workers,
                    a_io=self._io,
                    a_stop_event=self._stop_event,
                    a_id=i,
                    a_name=rel_path,
                    a_use_prof=self._use_prof,
                    a_use_cfg=self._use_cfg,
                    a_use_log=self._use_log,
                )
            else:
                raise ValueError(f"Unknown backend: {self._backend}")

            self._loaders[rel_path] = video_reader

    @property
    def loaders(self) -> BaseDict[str, OpenCVVideoCapture]:
        """Get the dictionary of video reader instances.

        Returns:
            BaseDict[str, Any]: Dictionary containing video readers.
        """
        return self._loaders

    def release(self) -> None:
        """Release all video readers in the data loader.

        This method calls the `release` method on each video reader instance
        to free up resources.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        for video_reader in self._loaders.values():
            video_reader.release()
        self._loaders.clear()

    def __len__(self) -> int:
        """Get the number of video loaders.

        Returns:
            int: The number of video loaders.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        return len(self._loaders)

    def __iter__(self) -> Iterator[Tuple[str, OpenCVVideoCapture]]:
        """Get an iterator over the video loaders.

        Returns:
            Iterator[Tuple[str, OpenCVVideoCapture]]: Iterator over video loader tuples.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        return iter(self._loaders.items())

    def __getitem__(self, a_key: str) -> OpenCVVideoCapture:
        """Get a video reader by its key.

        Args:
            a_key (str): The key of the video reader to retrieve.

        Returns:
            OpenCVVideoCapture: The video reader instance associated with the key.

        Raises:
            ValueError: If the dataset is not loaded or the key is not found.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        return self._loaders[a_key]
