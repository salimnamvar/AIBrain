"""Machine Learning - Object Tracking - Base Abstract Multi-Object Tracking Models

This module defines the `BaseTrkModel` class, which serves as an abstract base class
for multi-object tracking models. It provides a foundation for implementing specific
tracking models by defining common properties and methods.

Classes:
    BaseTrkModel: Abstract base class for multi-object tracking models.

Type Variables:
    IOT: Type variable for input/output handling, defaulting to `BaseIO`.
"""

from abc import ABC
from os import PathLike
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import openvino as ov

from aib.cnt.b_dict import BaseDict
from aib.cnt.io import BaseIO
from aib.cv.geom.box import AnyBox, AnyBoxList
from aib.cv.geom.size import IntSize
from aib.cv.img.frame import Frame2D
from aib.cv.img.image import Image2D
from aib.misc.common_types import StopEvent
from aib.ml.utils.b_ml_mdl import BaseMLModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseTrkModel(BaseMLModel[IOT], ABC):
    """Abstract Base Class for Multi-Object Tracking Models.

    This class provides a foundation for implementing multi-object tracking models.
    It defines common properties and methods that subclasses can extend or override.

    Attributes:
        src_ids (Optional[Tuple[int, ...]]): Source IDs for the tracking model.
        conf_thre (Optional[float]): Confidence threshold for detections.

    Methods:
        infer: Abstract method to perform inference. Must be implemented by subclasses.
    """

    def __init__(
        self,
        a_src_ids: Optional[Tuple[int, ...]] = None,
        a_conf_thre: Optional[float] = None,
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
        a_backend: Literal["ovms", "openvino", "sys", "opencv"] = "sys",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'BaseTrkModel',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the BaseTrkModel.

        Args:
            a_src_ids (Optional[Tuple[int, ...]]): Source IDs for the tracking model.
            a_conf_thre (Optional[float]): Confidence threshold for detections.
            a_model_uri (Optional[str | bytes | object | PathLike[str]]): URI of the model.
            a_model_version (Optional[int]): Version of the model.
            a_model_size (Optional[IntSize]): Size of the model.
            a_model_config (Optional[Dict[str, Any]]): Configuration dictionary for the model.
            a_model_in_layers (Optional[Tuple[str, ...]]): Input layer names for the model.
            a_model_out_layers (Optional[Tuple[str, ...]]): Output layer names for the model.
            a_backend_core (Optional[ov.Core | Any]): Backend core for inference.
            a_data_size (Optional[IntSize]): Size of the input data.
            a_infer_timeout (Optional[float]): Timeout for inference.
            a_infer_trial (int): Number of inference trials.
            a_device (Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"]): Device for inference.
            a_precision (Literal["FP32", "FP16", "INT8"]): Precision for inference.
            a_call_mode (Literal["sync", "async"]): Call mode for inference.
            a_io_mode (Literal["args", "queue", "ipc"]): IO mode for inference.
            a_proc_mode (Literal["batch", "online"]): Processing mode for inference.
            a_backend (Literal["ovms", "openvino", "sys", "opencv"]): Backend for inference.
            a_conc_mode (Optional[Literal["thread", "process"]]): Concurrency mode.
            a_max_workers (Optional[int]): Maximum number of workers.
            a_io (Optional[IOT]): IO object for the model.
            a_stop_event (Optional[StopEvent]): Stop event for the model.
            a_id (Optional[int]): ID of the model.
            a_name (str): Name of the model.
            a_use_prof (bool): Whether to use profiling.
            a_use_cfg (bool): Whether to use configuration.
            a_use_log (bool): Whether to use logging.
            **kwargs: Additional keyword arguments.
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
        self._src_ids: Optional[Tuple[int, ...]] = a_src_ids
        self._conf_thre: Optional[float] = a_conf_thre

    @property
    def src_ids(self) -> Optional[Tuple[int, ...]]:
        """Gets the source IDs for the tracking model.

        Returns:
            Optional[Tuple[int, ...]]: Source IDs.
        """
        return self._src_ids

    @property
    def conf_thre(self) -> Optional[float]:
        """Gets the confidence threshold for detections.

        Returns:
            Optional[float]: Confidence threshold.
        """
        return self._conf_thre

    def infer(
        self,
        *args: Any,
        a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
        a_boxes: Optional[AnyBoxList] = None,
        a_src_id: Optional[int] = None,
        a_step_timestamp: Optional[float] = None,
        a_step_id: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseDict[int, AnyBox]:
        """Performs inference on the input data.

        Args:
            a_image (Image2D | Frame2D | npt.NDArray[np.uint8]): Input image or frame.
            a_boxes (Optional[AnyBoxList]): List of bounding boxes.
            a_src_id (Optional[int]): Source ID.
            a_step_timestamp (Optional[float]): Timestamp of the step.
            a_step_id (Optional[int]): ID of the step.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseDict[int, AnyBox]: Dictionary of results.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement `infer` method.")
