"""Machine Learning - Object Detection - Base Abstract Models.

This module provides abstract base classes for object detection models, defining
common attributes, configuration options, and inference interfaces. These classes
serve as a foundation for implementing specific object detection architectures,
ensuring a consistent structure across synchronous and asynchronous execution modes.

Classes:
    BaseDetModel: Base abstract object detection model class with shared functionality.

Type Variables:
    IOT: Type variable bound to BaseIO for input/output handling.

Type Aliases:
    StopEvent: Union type for different event objects used to control execution termination.
"""

from abc import ABC
from os import PathLike
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import openvino as ov

from aib.cnt.io import BaseIO
from aib.cv.geom.box.bbox2d import AnyBBox2DList, AnyBBox2DNestedList
from aib.cv.geom.size import IntSize
from aib.cv.img.frame import Frame2D
from aib.cv.img.image import Image2D
from aib.misc.common_types import StopEvent
from aib.ml.utils.b_ml_mdl import BaseMLModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseDetModel(BaseMLModel[IOT], ABC):
    """Base abstract object detection model.

    Provides core functionality for object detection models, including thresholds,
    class filtering, and abstract inference methods. All object detection models
    should inherit from this class to ensure a consistent interface.

    Attributes:
        conf_thre (float): Confidence threshold for detections.
        nms_thre (float): Non-maximum suppression threshold.
        top_k_thre (int): Maximum number of detections to keep after NMS.
        min_size_thre (IntSize): Minimum bounding box size threshold.
        classes (Optional[Tuple[int, ...]]): Specific class IDs to filter detections.
        num_classes (Optional[int]): Total number of classes the model can predict.
        multi_label (bool): Indicates if multiple labels can be assigned to a single sample.

    Abstract Methods:
        infer(*args, a_images, **kwargs): Perform synchronous inference on a single image or a batch of images.
    """

    def __init__(
        self,
        a_conf_thre: Optional[float] = None,
        a_nms_thre: Optional[float] = None,
        a_top_k_thre: Optional[int] = None,
        a_min_size_thre: Optional[IntSize] = None,
        a_classes: Optional[Tuple[int, ...]] = None,
        a_num_classes: Optional[int] = None,
        a_multi_label: bool = False,
        a_model_uri: Optional[str | bytes | object | PathLike[str]] = None,
        a_model_version: Optional[int] = None,
        a_model_size: Optional[IntSize] = None,
        a_model_config: Optional[Dict[str, Any]] = None,
        a_model_in_layers: Optional[Tuple[str, ...]] = None,
        a_model_out_layers: Optional[Tuple[str, ...]] = None,
        a_backend_core: Optional[ov.Core | Any] = None,
        a_data_size: Optional[IntSize] = None,
        a_infer_timeout: Optional[float] = None,
        a_infer_trial: int = 1,
        a_device: Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"] = "AUTO",
        a_precision: Literal["FP32", "FP16", "INT8"] = "FP32",
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys", "opencv", "ultralytics"] = "sys",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'BaseDetModel',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ):
        """Initialize a BaseDetModel instance.

        Args:
            a_conf_thre (Optional[float], default=None):
                Confidence threshold for detections. Only predictions above this value
                are considered valid.
            a_nms_thre (Optional[float], default=None):
                Non-maximum suppression (NMS) threshold to remove overlapping detections.
            a_top_k_thre (Optional[int], default=None):
                Maximum number of detections to keep after applying NMS.
            a_min_size_thre (Optional[IntSize], default=None):
                Minimum bounding box size threshold for valid detections.
            a_classes (Optional[Sequence[int]], default=None):
                List of class IDs to filter detections. If None, all classes are considered.
            a_num_classes (Optional[int], default=None):
                Total number of classes the model can predict.
            a_multi_label (bool, default=False):
                Indicates if multiple labels can be assigned to a single sample.
            a_model_uri (Optional[str | bytes | object | PathLike[str]], default=None):
                Path, buffer, or object representing the model file to load.
            a_model_version (Optional[int]): Version number of the model. Defaults to None.
            a_model_size (Optional[IntSize], default=None):
                Expected input size for the model (e.g., image resolution).
            a_model_config (Optional[Dict[str, Any]], default=None):
                Runtime configuration options for the model.
            a_model_in_layers (Optional[Tuple[str]], optional):
                Names of the model input layers. Defaults to None.
            a_model_out_layers (Optional[Tuple[str]], optional):
                Names of the model output layers. Defaults to None.
            a_data_size (Optional[IntSize], default=None):
                Input data size (e.g., actual resolution of input data).
            a_infer_timeout (Optional[float], default=None):
                Timeout for inference execution, in seconds.
            a_infer_trial (int, default=1):
                Maximum number of inference retry attempts if a failure occurs.
            a_call_mode (Literal["sync", "async"], default="sync"):
                Execution mode of the model.
                - `"sync"`: Run inference synchronously.
                - `"async"`: Run inference asynchronously.
            a_io_mode (Literal["args", "queue", "ipc"], default="args"):
                Input/output communication mode.
                - `"args"`: Pass arguments directly.
                - `"queue"`: Use queue-based communication.
                - `"ipc"`: Use inter-process communication.
            a_proc_mode (Literal["batch", "online"], default="online"):
                Processing mode.
                - `"batch"`: Process data in batches.
                - `"online"`: Process data in real-time.
            a_backend (Literal["ovms", "openvino", "sys", "opencv", "ultralytics"], default="sys"):
                Backend used for inference.
            a_device (Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"], default="AUTO"):
                Target device for execution.
            a_precision (Literal["FP32", "FP16", "INT8"], optional):
                Model numerical precision. Defines the data type for weights, inputs, and outputs. Defaults to "FP32".
            a_conc_mode (Optional[Literal["thread", "process"]], default=None):
                Concurrency mode for parallel execution.
                - `"thread"`: Thread-based concurrency.
                - `"process"`: Process-based concurrency.
            a_max_workers (Optional[int], default=None):
                Maximum number of worker threads or processes.
            a_io (Optional[IOT], default=None):
                Input/output handler instance for managing data.
            a_stop_event (Optional[StopEvent], default=None):
                Event object for signaling stop conditions.
            a_id (Optional[int], default=None):
                Unique identifier for the model instance.
            a_name (str, default="BaseDetModel"):
                Model name used for logging and profiling.
            a_use_prof (bool, default=False):
                Whether to enable profiling of inference performance.
            a_use_cfg (bool, default=True):
                Whether to enable configuration management.
            a_use_log (bool, default=True):
                Whether to enable logging output.
            **kwargs (Any):
                Additional implementation-specific arguments.
        """
        super().__init__(
            a_model_uri=a_model_uri,
            a_model_version=a_model_version,
            a_model_size=a_model_size,
            a_model_config=a_model_config,
            a_model_in_layers=a_model_in_layers,
            a_model_out_layers=a_model_out_layers,
            a_backend_core=a_backend_core,
            a_data_size=a_data_size,
            a_infer_timeout=a_infer_timeout,
            a_infer_trial=a_infer_trial,
            a_device=a_device,
            a_precision=a_precision,
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
        self._conf_thre: Optional[float] = a_conf_thre
        self._nms_thre: Optional[float] = a_nms_thre
        self._top_k_thre: Optional[int] = a_top_k_thre
        self._min_size_thre: Optional[IntSize] = a_min_size_thre
        self._classes: Optional[Tuple[int, ...]] = a_classes
        self._num_classes: Optional[int] = a_num_classes
        self._multi_label: bool = a_multi_label

    @property
    def num_classes(self) -> Optional[int]:
        """Optional[int]: Total number of classes the model can predict."""
        return self._num_classes

    @property
    def multi_label(self) -> bool:
        """bool: Indicates if multiple labels can be assigned to a single sample."""
        return self._multi_label

    @property
    def conf_thre(self) -> Optional[float]:
        """Optional[float]: Confidence threshold for detections."""
        return self._conf_thre

    @property
    def nms_thre(self) -> Optional[float]:
        """Optional[float]: Non-maximum suppression threshold."""
        return self._nms_thre

    @property
    def top_k_thre(self) -> Optional[int]:
        """Optional[int]: Maximum number of detections to keep after NMS."""
        return self._top_k_thre

    @property
    def min_size_thre(self) -> Optional[IntSize]:
        """Optional[IntSize]: Minimum bounding box size threshold."""
        return self._min_size_thre

    @property
    def classes(self) -> Optional[Tuple[int, ...]]:
        """Optional[Tuple[int, ...]]: Class IDs to filter detections."""
        return self._classes

    def infer(
        self,
        *args: Any,
        a_images: Union[
            Image2D,
            Frame2D,
            npt.NDArray[np.uint8],
            Sequence[Image2D],
            Sequence[Frame2D],
            Sequence[npt.NDArray[np.uint8]],
        ],
        **kwargs: Any,
    ) -> AnyBBox2DList | AnyBBox2DNestedList:
        """Perform synchronous inference on one or more images.

        Args:
            *args: Additional positional arguments for inference.
            a_images (Union[Image2D, Frame2D, npt.NDArray[np.uint8], Sequence[Image2D], Sequence[Frame2D], Sequence[npt.NDArray[np.uint8]]]):
                Input image(s) for detection.
            **kwargs: Additional keyword arguments for inference.

        Returns:
            AnyBBox2DList | AnyBBox2DNestedList: Detected bounding boxes with class IDs and confidence scores.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement `infer` method.")
