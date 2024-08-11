""" Camera List

    This Python file contains the definition of the :class:`CameraList` class, an object class that stores a
    collection of :class:`Camera` objects.
"""

# region Imported Dependencies
import concurrent.futures as cf
import logging
import time
import traceback
import uuid
import warnings
from typing import List

from brain.util.cfg import BrainConfig
from brain.util.cv.cam import Camera
from brain.util.cv.vid import Frame2DList
from brain.util.misc import BaseHealthStatus
from brain.util.obj import BaseObjectList


# endregion Imported Dependencies


class CameraListHealthStatus(BaseHealthStatus):
    """Camera List Health Status

    This class defines the health status for a :class:`CameraList`, including specific indicators for tracking streaming
    errors.

    Attributes:
        NAME (str): A string representing the name of the CameraList health status.
        STREAM_ERROR (bool): A boolean indicating whether there is an error during the streaming process.
    """

    NAME: str = "CameraList"
    STREAM_ERROR: bool = False


class CameraList(BaseObjectList[Camera]):
    """Camera List

    The CameraList class is based on the :class:`ObjectList` class and serves as a container for a collection of
    :class:`Camera` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the CameraList (default is 'CameraList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[Camera], optional):
            A list of Camera objects to initialize the CameraList (default is None).
        cfg: BrainConfig
            An instance of the BrainConfig class for configuration settings.
        logger: logging.Logger
            A :class:`logging.Logger` instance for logging camera-list-related messages.
        frames: Frame2DList
            A :class:`Frame2DList` for storing :class:`Frame2D` as the latest frames from the cameras. From each camera
            one frame will be kept in this list.
        warmup_duration: int
            An int value that indicates the warmup duration (in second) that the process of streaming can freeze to
            let the cameras for reading their first frame.
    """

    def __init__(
        self,
        a_name: str = "CameraList",
        a_max_size: int = -1,
        a_items: List[Camera] = None,
        a_warmup_duration: int = 1,
    ):
        """
        Constructor for the `CameraList` class.

        Args:
            a_name (str, optional):
                A :type:`string` that specifies the name of the `CameraList` instance (default is 'CameraList').
            a_max_size (int, optional):
                An :type:`int` representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[Camera], optional):
                A list of :class:`Camera` objects to initialize the `CameraList` (default is None).
            a_warmup_duration (int, optional):
                An int value that indicates the warmup duration (in second) that the process of streaming can freeze to
                let the
                cameras for reading their first frame.

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.logger = logging.getLogger(self.cfg.log.name + "." + self.name)
        self.health_status: CameraListHealthStatus = CameraListHealthStatus(a_name)
        self.warmup_duration: int = a_warmup_duration

    def stream(self) -> None:
        """Stream Method

        This method handles the streaming process of all cameras in the list by creating a specific thread for each one
        and running them concurrently.

        The method utilizes a thread pool executor to concurrently run the stream method of each camera. It includes a
        mechanism to wait until at least one camera is successfully streaming before proceeding. If no camera is
        streaming within a certain timeframe, a warning message is logged, and a warning is raised.

        Returns:
            None: The method does not return any values.

        Raises:
            RuntimeError: If an error occurs during the streaming process.
            Warning: If no camera is successfully streaming within the specified timeframe.
        """

        try:
            executor = cf.ThreadPoolExecutor()

            # Map each camera's stream method to a thread in the thread pool
            futures = [executor.submit(camera.stream) for camera in self.items]

            # Wait until at least one camera is read
            while len(self.frames) < 1:
                msg = f"{self.name} is trying to at least stream one camera."
                self.logger.warning(msg)
                warnings.warn(msg)
                time.sleep(self.warmup_duration)
            self.health_status.STREAM_ERROR = False
        except Exception as e:
            self.health_status.STREAM_ERROR = True
            msg = f"{self.name}'s `stream` method got an error of `{e}`."
            self.logger.fatal(msg)
            error = traceback.format_exc()
            self.logger.fatal(error)
            raise RuntimeError(msg)

    @property
    def frames(self) -> Frame2DList:
        """Frames Property

        This property provides a list of :class:`Frame2D` objects as a :class:`Frame2DList` containing the latest frame
        from each camera.

        Returns:
            Frame2DList: A :class:`Frame2DList` object containing the latest frames from all cameras.
        """
        return Frame2DList(
            a_items=[camera.frame for camera in self.items if len(camera.frames) > 0]
        )

    def get_ids(self) -> List[uuid.UUID]:
        """Get IDs of all cameras.

        Returns a list of UUIDs extracted from the :class:`Camera` objects in the list.

        Returns:
            List[uuid.UUID]: A list of UUIDs.
        """
        return [item.id for item in self.items]
