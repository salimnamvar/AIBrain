"""Machine Learning - Base Abstract Machine Learning Models

This module defines the base abstract machine learning model classes for the system, providing a foundation
for machine learning model implementations.

Classes:
    BaseMLModel: Base abstract machine learning model class with common functionality.

Type Variables:
    - IOT: Type variable bound to BaseIO for input/output handling.

Type Aliases:
    - StopEvent: Union type for different event objects used in stopping execution.
"""

from abc import ABC
from os import PathLike
from typing import Any, Callable, Dict, Literal, Optional, Tuple, TypeVar

import grpc
import openvino as ov
import ultralytics as ul
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from src.utils.cnt.io import BaseIO
from src.utils.cv.geom.size import IntSize
from src.utils.misc.common_types import StopEvent
from src.utils.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseMLModel(BaseModel[IOT], ABC):
    """Base Abstract Machine Learning Model Class

    This class provides a foundation for implementing machine learning models
    in the system. It defines shared attributes, configuration handling,
    and abstract methods that subclasses must implement.

    Attributes:
        model (Optional[Any]): The machine learning model object.
        model_uri (Optional[str | bytes | object | PathLike[str]]):
            Path/URI or object representing the model source.
        model_version (Optional[int]): Version number of the model.
        model_size (Optional[IntSize]): Expected input size of the model (width, height).
        model_config (Optional[Dict[str, Any]]): Runtime configuration dictionary.
        model_in_layers (Optional[Tuple[str]]): Names of the model input layers.
        model_out_layers (Optional[Tuple[str]]): Names of the model output layers.
        backend_core (Optional[ov.Core | Any]): Core object used for compiling and managing models such as OpenVINO.
        data_size (Optional[IntSize]): Expected input data size for processing.
        device (str): Target device for inference (e.g., "CPU", "GPU").
        precision (Literal["FP32", "FP16", "INT8"]): Numerical precision for the model.
        infer_timeout (Optional[float]): Maximum time allowed for inference.
        infer_trial (int): Maximum number of inference attempts on failure.
        grpc_channel (Optional[grpc.Channel | grpc.aio.Channel]): gRPC channel
            for communication with OVMS.
        service_stub (Optional[PredictionServiceStub]): gRPC stub for making prediction requests.

    Abstract Methods:
        preproc(*args, **kwargs): Define preprocessing logic before inference.
        postproc(*args, **kwargs): Define postprocessing logic after inference.
        train(*args, **kwargs): Define model training procedure.
        test(*args, **kwargs): Define model testing procedure.
        create_infer_request(*args, **kwargs): Define inference request creation procedure.
    """

    def __init__(
        self,
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
        a_name: str = 'BaseMLModel',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a base machine learning model.

        Args:
            a_model_uri (Optional[str | bytes | object | PathLike[str]], optional):
                Path/URI to the model source or an in-memory model object. Defaults to None.
            a_model_version (Optional[int]): Version number of the model. Defaults to None.
            a_model_size (Optional[IntSize], optional):
                Expected input size for the model (width, height). Defaults to None.
            a_model_config (Optional[Dict[str, Any]], optional):
                Dictionary of runtime configuration options (e.g., thresholds, hyperparameters). Defaults to None.
            a_model_in_layers (Optional[Tuple[str]], optional):
                Names of the model input layers. Defaults to None.
            a_model_out_layers (Optional[Tuple[str]], optional):
                Names of the model output layers. Defaults to None.
            a_backend_core (Optional[ov.Core | Any]): Core instance for compilation and runtime such as OpenVINO.
            a_data_size (Optional[IntSize], optional):
                Input data size expected by the model pipeline (may differ from model size). Defaults to None.
            a_infer_timeout (Optional[float], optional):
                Maximum time allowed for inference in seconds. Defaults to None.
            a_infer_trial (int, optional):
                Maximum number of inference attempts on failure. Defaults to 1.
            a_call_mode (Literal["sync", "async"], optional):
                Subsystem execution mode: synchronous or asynchronous. Defaults to "sync".
            a_io_mode (Literal["args", "queue", "ipc"], optional):
                I/O handling mode for passing data ("args", "queue", or "ipc"). Defaults to "args".
            a_proc_mode (Literal["batch", "online"], optional):
                Processing mode: batch or online/streaming. Defaults to "online".
            a_backend (Literal["ovms", "openvino", "sys", "opencv", "ultralytics"], optional):
                Backend engine to use for inference. Defaults to "sys".
            a_conc_mode (Optional[Literal["thread", "process"]], optional):
                Concurrency mode for running jobs. Defaults to None.
            a_max_workers (Optional[int], optional):
                Maximum number of concurrent workers. Defaults to None.
            a_device (Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"], optional):
                Target device for execution. Defaults to "AUTO".
            a_precision (Literal["FP32", "FP16", "INT8"], optional):
                Model numerical precision. Defines the data type for weights, inputs, and outputs. Defaults to "FP32".
            a_io (Optional[IOT], optional):
                Input/output interface object for data handling. Defaults to None.
            a_stop_event (Optional[StopEvent], optional):
                Event object to signal stopping of the subsystem. Defaults to None.
            a_id (Optional[int], optional):
                Unique identifier for the subsystem instance. Defaults to None.
            a_name (str, optional):
                Human-readable name for the subsystem instance. Defaults to "BaseMLModel".
            a_use_prof (bool, optional):
                Enable profiling metrics such as FPS or iteration timing. Defaults to False.
            a_use_cfg (bool, optional):
                Enable configuration management. Defaults to True.
            a_use_log (bool, optional):
                Enable logging for debugging and runtime information. Defaults to True.
            **kwargs (Any):
                Additional keyword arguments forwarded to BaseModel.
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
        self._model: Optional[Callable[..., Any] | ul.YOLO | ov.CompiledModel] = None
        self._model_uri: Optional[str | bytes | object | PathLike[str]] = a_model_uri
        self._model_version: Optional[int] = a_model_version
        self._model_size: Optional[IntSize] = a_model_size
        self._model_in_layers: Optional[Tuple[str, ...]] = a_model_in_layers
        self._model_out_layers: Optional[Tuple[str, ...]] = a_model_out_layers
        self._data_size: Optional[IntSize] = a_data_size
        self._device: Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"] = a_device
        self._precision: Literal["FP32", "FP16", "INT8"] = a_precision
        self._infer_timeout: Optional[float] = a_infer_timeout
        self._infer_trial: int = a_infer_trial
        self._model_config: Optional[Dict[str, Any]] = a_model_config
        self._grpc_channel: Optional[grpc.Channel | grpc.aio.Channel] = None
        self._service_stub: Optional[PredictionServiceStub] = None
        self._backend_core: Optional[ov.Core] = a_backend_core

    @property
    def model_in_layers(self) -> Optional[Tuple[str, ...]]:
        """Get the model input layer names.

        Returns:
            Optional[Tuple[str]]: The model input layer names.
        """
        return self._model_in_layers

    @property
    def model_out_layers(self) -> Optional[Tuple[str, ...]]:
        """Get the model output layer names.

        Returns:
            Optional[Tuple[str]]: The model output layer names.
        """
        return self._model_out_layers

    @property
    def precision(self) -> str:
        """Get the numerical precision for the model.

        Returns:
            str: The precision type (e.g., 'float32', 'float16', 'int8').
        """
        return self._precision

    @property
    def model_version(self) -> Optional[int]:
        """Get the model version.

        Returns:
            Optional[int]: The model version.
        """
        return self._model_version

    @property
    def grpc_channel(self) -> Optional[grpc.Channel | grpc.aio.Channel]:
        """Get the gRPC channel for communication with the OVMS server.

        Returns:
            Optional[grpc.Channel | grpc.aio.Channel]: The gRPC channel.
        """
        return self._grpc_channel

    @property
    def service_stub(self) -> Optional[PredictionServiceStub]:
        """Get the prediction service stub for communication with the OVMS server.

        Returns:
            Optional[PredictionServiceStub]: The prediction service stub.
        """
        return self._service_stub

    @property
    def model_config(self) -> Optional[Dict[str, Any]]:
        """Get the runtime configuration options.

        Returns:
            Optional[Dict[str, Any]]: The runtime configuration options.
        """
        return self._model_config

    @property
    def infer_timeout(self) -> Optional[float]:
        """Get the inference timeout.

        Returns:
            Optional[float]: The inference timeout.
        """
        return self._infer_timeout

    @property
    def infer_trial(self) -> int:
        """Get the maximum number of inference attempts on failure.

        Returns:
            int: Number of trials for inference. If an inference fails, it will be retried
                up to this number of times.

        Raises:
            AttributeError: If inference trial is not set.
        """
        return self._infer_trial

    @property
    def device(self) -> str:
        """Get the target device.

        Returns:
            str: The target device (e.g., "CPU", "GPU").
        """
        return self._device

    @property
    def data_size(self) -> Optional[IntSize]:
        """Get the input data size.

        Returns:
            Optional[IntSize]: The input data size.
        """
        return self._data_size

    @property
    def model_size(self) -> Optional[IntSize]:
        """Get the input size of the model.

        Returns:
            Optional[IntSize]: Model input size (width, height).
        """
        return self._model_size

    @property
    def model(self) -> Optional[Callable[..., Any] | ul.YOLO | ov.CompiledModel]:
        """Model Property

        This property returns the machine learning model object.

        Returns:
            Any: The machine learning model object.
        """
        return self._model

    @property
    def model_uri(self) -> Optional[str | bytes | object | PathLike[str]]:
        """Model Source Path or Object Property

        This property returns the model source or object associated with the machine learning model.

        Returns:
            str | bytes | object | Path: The model source or object.
        """
        return self._model_uri

    @property
    def backend_core(self) -> Optional[ov.Core | Any]:
        """Get the backend core instance.

        Returns:
            Optional[ov.Core]: The backend core instance.
        """
        return self._backend_core

    def preproc(self, *args: Any, **kwargs: Any) -> Any:
        """Preprocessing Method

        This method is intended to be overridden by subclasses to implement specific preprocessing logic.
        It should handle any necessary data preparation before the model's main processing step.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for preprocessing.
            **kwargs (Dict[str, Any]): Keyword arguments for preprocessing.

        Returns:
            Any: The preprocessed data.
        """
        raise NotImplementedError(f"`{self.name}` must implement `_preproc`")

    def postproc(self, *args: Any, **kwargs: Any) -> Any:
        """Postprocessing Method

        This method is intended to be overridden by subclasses to implement specific postprocessing logic.
        It should handle any necessary data transformation after the model's main processing step.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for postprocessing.
            **kwargs (Dict[str, Any]): Keyword arguments for postprocessing.

        Returns:
            Any: The postprocessed data.
        """
        raise NotImplementedError(f"`{self.name}` must implement `_postproc`")

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Training Method

        This method is intended to be overridden by subclasses to implement specific training logic.
        It should handle the training process of the model.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for training.
            **kwargs (Dict[str, Any]): Keyword arguments for training.

        Returns:
            None: This method does not return any value.
        """
        raise NotImplementedError(f"`{self.name}` must implement `train`")

    def test(self, *args: Any, **kwargs: Any) -> None:
        """Testing Method

        This method is intended to be overridden by subclasses to implement specific testing logic.
        It should handle the testing process of the model.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for testing.
            **kwargs (Dict[str, Any]): Keyword arguments for testing.

        Returns:
            None: This method does not return any value.
        """
        raise NotImplementedError(f"`{self.name}` must implement `test`")

    def create_infer_request(self, *args: Any, **kwargs: Any) -> Any:
        """Create Inference Request Method

        This method is intended to be overridden by subclasses to implement specific inference request creation logic.
        It should handle the preparation of the inference request for the model.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for creating the inference request.
            **kwargs (Dict[str, Any]): Keyword arguments for creating the inference request.

        Returns:
            Any: The created inference request.
        """
        raise NotImplementedError(f"`{self.name}` must implement `create_infer_request`")

    def decode_preds(self, *args: Any, **kwargs: Any) -> Any:
        """Decode Predictions Method

        This method is intended to be overridden by subclasses to implement specific predictions decoding logic.
        It should handle the transformation of the model's predictions into a usable format.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for decoding the predictions.
            **kwargs (Dict[str, Any]): Keyword arguments for decoding the predictions.

        Returns:
            Any: The decoded predictions.
        """
        raise NotImplementedError(f"`{self.name}` must implement `decode_preds`")
