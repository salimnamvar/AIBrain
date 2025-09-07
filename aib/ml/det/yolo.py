"""
Machine Learning - Object Detection - YOLO Models

This module provides a YOLO object detection model wrapper supporting
OpenVINO Runtime (OV) and OpenVINO Model Server (OVMS) backends. It includes
preprocessing, postprocessing, synchronous and asynchronous inference,
and queue-based I/O support.

Classes:
    YOLO: Wrapper class for YOLO object detection models, supporting
        synchronous and asynchronous inference modes with OpenVINO or OVMS.

Type Variables:
    IOT: Type variable representing the input/output queue interface used
        for asynchronous inference. It defaults to a QueueIO handling
        Frame2D inputs and bounding box outputs.
"""

import asyncio
import warnings
from os import PathLike
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, Union, cast

import cv2
import grpc
import numpy as np
import numpy.typing as npt
import openvino as ov
import torch
import ultralytics as ultra
from tensorflow import make_ndarray, make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from ultralytics.engine.results import Results as UltraResults

from src.utils.cnt.io import QueueIO
from src.utils.cv.geom.box.bbox2d import IntBBox2DList, IntBBox2DNestedList
from src.utils.cv.geom.size import IntSize
from src.utils.cv.img.frame import Frame2D
from src.utils.cv.img.image import Image2D
from src.utils.misc.common_types import StopEvent
from src.utils.ml.det.utils.b_det_mdl import BaseDetModel
from src.utils.ml.det.utils.nms import UltralyticsNMS
from src.utils.ml.det.utils.preproc import letterbox

IOT = TypeVar(
    "IOT",
    bound=Union[
        QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DList]],
        QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DNestedList]],
    ],
    default=QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DList]],
)


class YOLO(BaseDetModel[IOT]):
    """
    YOLO object detection model wrapper supporting OpenVINO and OVMS backends.

    This class provides a unified interface for loading, preprocessing,
    performing inference, and postprocessing for YOLO-based object detection models.
    It supports synchronous and asynchronous inference modes, multiple backends
    (OpenVINO Runtime and OVMS), and various I/O configurations.

    Features:
        - Backend-agnostic inference with OpenVINO or OVMS.
        - Preprocessing and postprocessing utilities tailored for YOLO models.
        - Support for synchronous (`infer`) and asynchronous (`infer_async`, `run_async`) pipelines.
        - Flexible input formats: single image/frame, sequence of images/frames, or NumPy arrays.
        - Configurable thresholds for confidence, NMS, and top-K predictions.
        - Optional multi-label detection and class filtering.
        - Async queue-based I/O with concurrency and profiling support.

    Attributes:
        id (Optional[int]): Unique identifier for the object instance.
        name (str): Human-readable name of the object (default: 'YOLOFireDetector').
        cfg (Optional[Configuration]): Configuration instance if enabled.
        logger (Optional[logging.Logger]): Logger instance if logging is enabled.
        use_prof (bool): Flag to enable profiling (default: False).
        use_cfg (bool): Flag to enable configuration access (default: True).
        use_log (bool): Flag to enable logging (default: True).
        init_time (float): Time when the object was initialized.
        profiler (Optional[Profiler]): Profiler instance for performance monitoring.
        iter (int): Current iteration count of the model.
        call_mode (str): Execution mode, either "sync" or "async" (default: "sync").
        io_mode (str): Input feeding mode: "args", "queue", or "ipc" (default: "args").
        proc_mode (str): Processing mode, either "batch" or "online" (default: "online").
        backend (str): Backend engine for model execution: "openvino" or "ovms" (default: "openvino").
        io (Optional[IOT]): IO object for feeding or retrieving data.
        stop_event (Optional[StopEvent]): Stop event for graceful shutdown.
        conc_mode (Optional[str]): Concurrency mode: "thread" or "process".
        max_workers (Optional[int]): Maximum number of concurrent workers.
        executor (Optional[ProcessPoolExecutor | ThreadPoolExecutor]): Executor for parallel jobs.
        model (Optional[Any]): The machine learning model object.
        model_uri (Optional[str | bytes | object | PathLike[str]]): Path/URI or object representing the model source.
        model_version (Optional[int]): Version number of the model.
        model_size (Optional[IntSize]): Expected input size of the model (width, height).
        model_config (Optional[Dict[str, Any]]): Runtime configuration dictionary.
        model_in_layers (Optional[Tuple[str, ...]]): Names of the model input layers (default: ("images",)).
        model_out_layers (Optional[Tuple[str, ...]]): Names of the model output layers (default: ("output0",)).
        data_size (Optional[IntSize]): Expected input data size for processing.
        device (str): Target device for inference: "CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO" (default: "AUTO").
        precision (str): Numerical precision for the model: "FP32", "FP16", "INT8" (default: "FP32").
        infer_timeout (Optional[float]): Maximum time allowed for inference in seconds.
        infer_trial (int): Maximum number of inference attempts on failure (default: 1).
        grpc_channel (Optional[grpc.Channel | grpc.aio.Channel]): gRPC channel for communication with OVMS.
        service_stub (Optional[PredictionServiceStub]): gRPC stub for making prediction requests.
        core (Optional[ov.Core]): OpenVINO Core object used for compiling and managing models.
        conf_thre (Optional[float]): Confidence threshold for filtering detections.
        nms_thre (Optional[float]): Non-maximum suppression (NMS) threshold for removing overlapping detections.
        top_k_thre (Optional[int]): Maximum number of detections to keep after applying NMS.
        min_size_thre (Optional[IntSize]): Minimum bounding box size threshold for valid detections.
        classes (Optional[Tuple[int, ...]]): Specific class IDs to filter detections.
        num_classes (Optional[int]): Total number of classes the model can predict.
        multi_label (bool): Indicates if multiple labels can be assigned to a single sample (default: False).

    Methods:
        load(): Load the model for inference depending on the backend.
        preproc(a_images, a_model_size): Preprocess input images into model-ready format.
        postproc(a_dets, a_logits, a_data_sizes, a_top_k_thre, a_conf_thre, a_proc_mode):
            Postprocess model outputs into bounding boxes.
        create_infer_request(a_images, a_model_name, a_model_in_layers, a_model_version):
            Create a gRPC PredictRequest for OVMS.
        decode_preds(a_response, a_model_out_layers): Decode OVMS PredictResponse into numpy arrays.
        infer(a_images): Perform synchronous inference.
        infer_async(): Perform asynchronous inference on input frames.
        run_async(): Run multiple asynchronous inference workers concurrently.

    Usage:
        1. Initialize the model instance with desired configuration.
        2. Call `load()` to load and compile the model (OpenVINO) or setup OVMS connection.
        3. Perform inference:
            - Synchronously using `infer(images)`.
            - Asynchronously using `infer_async()` or `run_async()` with queue-based I/O.
        4. Preprocessing (`preproc`) and postprocessing (`postproc`) are provided as class methods
           to handle input resizing, normalization, and bounding box scaling/NMS.
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
        a_model_size: Optional[IntSize] = IntSize(width=640, height=640),
        a_model_config: Optional[Dict[str, Any]] = None,
        a_model_in_layers: Optional[Tuple[str, ...]] = ("images",),
        a_model_out_layers: Optional[Tuple[str, ...]] = ("output0",),
        a_backend_core: Optional[ov.Core | Any] = None,
        a_data_size: Optional[IntSize] = None,
        a_infer_timeout: Optional[float] = None,
        a_infer_trial: int = 1,
        a_device: Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"] = "AUTO",
        a_precision: Literal["FP32", "FP16", "INT8"] = "FP32",
        a_call_mode: Literal["sync", "async"] = "sync",
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys", "ultralytics"] = "ultralytics",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'YOLO',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize a YOLO object detection model instance.

        Args:
            a_conf_thre (float, optional): Confidence threshold for filtering detections.
            a_nms_thre (float, optional): Intersection-over-Union (IoU) threshold for Non-Maximum Suppression.
            a_top_k_thre (int, optional): Maximum number of top predictions to keep.
            a_min_size_thre (IntSize, optional): Minimum size threshold for detected objects.
            a_classes (Tuple[int, ...], optional): Specific classes to detect. Defaults to all classes.
            a_num_classes (int, optional): Total number of classes in the model.
            a_multi_label (bool): Whether multiple labels per detection box are allowed. Defaults to False.
            a_backend_core (ov.Core, optional): OpenVINO Runtime Core object. Required if backend is "openvino".
            a_model_uri (str | bytes | object | PathLike[str], optional):
                Path to the model file or OVMS server URI (host:port).
            a_model_version (int, optional): Model version to load (used for OVMS).
            a_model_size (IntSize, optional): Input dimensions for the model. Defaults to 640x640.
            a_model_config (Dict[str, Any], optional): Backend-specific configuration parameters.
            a_model_in_layers (Tuple[str, ...], optional): Names of model input layers. Defaults to ("images",).
            a_model_out_layers (Tuple[str, ...], optional): Names of model output layers. Defaults to ("output0",).
            a_data_size (IntSize, optional): Original input data size for scaling postprocessing.
            a_infer_timeout (float, optional): Timeout for inference requests (seconds).
            a_infer_trial (int): Number of retries for inference if it fails. Defaults to 1.
            a_device (str):
                Device for inference. Options: "CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO". Defaults to "AUTO".
            a_precision (str): Model precision: "FP32", "FP16", or "INT8". Defaults to "FP32".
            a_call_mode (str): Inference call mode: "sync" or "async". Defaults to "sync".
            a_io_mode (str): I/O handling mode: "args", "queue", or "ipc". Defaults to "args".
            a_proc_mode (str): Processing mode: "batch" or "online". Defaults to "online".
            a_backend (str): Backend type: "openvino", "ovms", or "sys". Defaults to "openvino".
            a_conc_mode (str, optional): Concurrency mode: "thread" or "process". Required for async queue I/O.
            a_max_workers (int, optional): Maximum number of concurrent workers for async processing.
            a_io (IOT, optional): Input/output queue interface object.
            a_stop_event (StopEvent, optional): Event object to stop async processing gracefully.
            a_id (int, optional): Identifier for the model instance or version.
            a_name (str): Model name. Defaults to 'YOLOFireDetector'.
            a_use_prof (bool): Enable profiling. Defaults to False.
            a_use_cfg (bool): Enable configuration usage. Defaults to True.
            a_use_log (bool): Enable logging. Defaults to True.
            **kwargs: Additional keyword arguments passed to the base class.

        Notes:
            This method initializes all parameters, backend settings, I/O configuration, and
            inference options. The actual model loading is done in the `load()` method.
        """
        super().__init__(
            a_conf_thre=a_conf_thre,
            a_nms_thre=a_nms_thre,
            a_top_k_thre=a_top_k_thre,
            a_min_size_thre=a_min_size_thre,
            a_classes=a_classes,
            a_num_classes=a_num_classes,
            a_multi_label=a_multi_label,
            a_backend_core=a_backend_core,
            a_model_uri=a_model_uri,
            a_model_version=a_model_version,
            a_model_size=a_model_size,
            a_model_config=a_model_config,
            a_model_in_layers=a_model_in_layers,
            a_model_out_layers=a_model_out_layers,
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

    def load(self) -> None:
        """
        Load the YOLO model according to the specified backend.

        This method initializes the model for inference. Depending on the backend,
        it either loads and compiles the model using OpenVINO Runtime or sets up
        a gRPC connection to an OVMS (OpenVINO Model Server) instance.

        Raises:
            AssertionError: If required parameters for the selected backend are missing.
            NotImplementedError: If the specified backend is not supported.
            RuntimeError: If model loading fails for any other reason.

        Backend behavior:
            - openvino:
                - Requires `self.core` and `self.device` to be set.
                - Reads the model from `self.model_uri`.
                - If `self.model_size` is not set, it infers input size from the model.
                - Compiles the model with OpenVINO Runtime.
            - ovms:
                - `self.model_uri` must be a string in the format "host:port".
                - Creates a gRPC channel (synchronous or asynchronous based on `self.call_mode`).
                - Initializes the prediction service stub for sending inference requests.

        Notes:
            The actual inference is performed by the `infer()` or `infer_async()` methods
            after loading the model.
        """
        try:
            assert self.model_uri is not None, "Model URI must be set."

            if self.backend == "openvino":
                assert self.backend_core is not None, "OpenVINO Runtime Core must be set."
                assert self.device is not None, "Device must be set."

                model = self.backend_core.read_model(self.model_uri)
                if self.model_size is None:
                    input_layer = model.input(0)
                    _, _, height, width = input_layer.shape
                    self._model_size = IntSize(width=int(width), height=int(height))
                self._model = self.backend_core.compile_model(model, self.device, self.model_config)

            elif self.backend == "ovms":
                assert isinstance(self.model_uri, str), "Model URI must be a string in the format of `host:port`."

                if self.call_mode == 'sync':
                    self._grpc_channel = grpc.insecure_channel(self.model_uri)
                elif self.call_mode == 'async':
                    self._grpc_channel = grpc.aio.insecure_channel(self.model_uri)
                self._service_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel)

            elif self.backend == "ultralytics":
                self._model = ultra.YOLO(model=self.model_uri)

            else:
                raise NotImplementedError(
                    f"Loading models for backend `{self.backend}` is not implemented. "
                    f"Supported backends: `openvino`, `ovms`, `ultralytics`."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from: {self.model_uri}. Original error: {e}") from e

    @classmethod
    def preproc(
        cls,
        a_images: Union[
            Image2D,
            Frame2D,
            npt.NDArray[Any],
            Sequence[Image2D],
            Sequence[Frame2D],
            Sequence[npt.NDArray[Any]],
        ],
        a_model_size: IntSize,
        a_backend: Literal["openvino", "ovms", "sys", "ultralytics"],
        a_precision: Literal["FP32", "FP16", "INT8"],
    ) -> Union[List[npt.NDArray[Any]], Tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """
        Preprocess input images for YOLO model inference.

        This method resizes, normalizes, and formats input images to match the
        model's expected input size and precision. It supports single images
        or sequences of images in different formats (Image2D, Frame2D, or NumPy arrays).

        Args:
            a_images (Union[Image2D, Frame2D, np.ndarray, Sequence[Image2D], Sequence[Frame2D], Sequence[np.ndarray]]):
                Input image(s) to preprocess.
            a_model_size (IntSize): Target model input size (width, height).
            a_precision (Literal["FP32", "FP16", "INT8"], optional):
                Precision for the output arrays. Defaults to "FP32".
                - FP32: Floating point 32-bit normalized to [0,1].
                - FP16: Floating point 16-bit normalized to [0,1].
                - INT8: 8-bit integer.

        Returns:
            Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
                - Preprocessed images as a NumPy array of shape `(batch_size, channels, height, width)`.
                - Original sizes of the images as a NumPy array of shape `(batch_size, 2)`.

        Raises:
            RuntimeError: If preprocessing fails for any image.
        """

        def pre_transform(
            a_image: npt.NDArray[np.uint8], a_target_size: IntSize
        ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
            orig_h, orig_w, _ = a_image.shape
            data_size = np.array([[orig_h, orig_w]], dtype=np.float32)
            image = letterbox(a_image=a_image, a_target_size=a_target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
            if a_precision == 'FP32':
                image = image.astype(np.float32) / 255.0
                data_size = data_size.astype(np.float32)
            elif a_precision == 'FP16':
                image = image.astype(np.float16) / 255.0
                data_size = data_size.astype(np.float16)
            elif a_precision == 'INT8':
                image = image.astype(np.int8)
                data_size = data_size.astype(np.int8)
            return image, data_size

        try:
            if a_backend == "ultralytics":
                images: List[npt.NDArray[Any]] = []
                if isinstance(a_images, (list, tuple)):
                    for image in a_images:
                        if isinstance(image, (Image2D, Frame2D)):
                            image = image.data
                        images.append(image)
                else:
                    if isinstance(a_images, (Image2D, Frame2D)):
                        image = a_images.data
                    else:
                        image = cast(npt.NDArray[Any], a_images)
                    images = [image]
                return images

            if isinstance(a_images, (list, tuple)):
                images: List[npt.NDArray[Any]] = []
                data_sizes: List[npt.NDArray[Any]] = []
                for image in a_images:
                    if isinstance(image, (Image2D, Frame2D)):
                        image = image.data
                    image, size = pre_transform(image, a_target_size=a_model_size)
                    images.append(image)
                    data_sizes.append(size)
            else:
                if isinstance(a_images, (Image2D, Frame2D)):
                    image = a_images.data
                else:
                    image = a_images
                image, size = pre_transform(cast(npt.NDArray[np.uint8], image), a_model_size)
                images = [image]
                data_sizes = [size]

            return np.concatenate(images, axis=0), np.concatenate(data_sizes, axis=0)

        except Exception as e:
            raise RuntimeError(f"Preprocessing failed in {cls.__name__}. Original error: {e}") from e

    @classmethod
    def postproc(
        cls,
        a_preds: npt.NDArray[Any] | List[UltraResults],
        a_proc_mode: Literal["batch", "online"],
        a_backend: Literal["openvino", "ovms", "sys", "ultralytics"],
        a_data_sizes: Optional[npt.NDArray[Any]] = None,
        a_model_size: Optional[IntSize] = None,
        a_conf_thre: Optional[float] = None,
        a_nms_thre: Optional[float] = None,
        a_top_k_thre: Optional[int] = None,
        a_classes: Optional[Tuple[int, ...]] = None,
        a_num_classes: Optional[int] = None,
        a_multi_label: bool = False,
    ) -> IntBBox2DList | IntBBox2DNestedList:
        """
        Postprocess raw predictions from a YOLO model.

        This method applies Non-Maximum Suppression (NMS), rescales bounding
        boxes to the original image dimensions, and organizes the predictions
        into structured bounding box lists. It supports both batch and online
        processing modes.

        Args:
            a_preds (np.ndarray): Raw model predictions of shape `(batch_size, num_predictions, 5+num_classes)`.
            a_data_sizes (np.ndarray): Original image sizes of shape `(batch_size, 2)` containing height and width.
            a_model_size (IntSize): Model input size (width, height) used for scaling the boxes back to original size.
            a_conf_thre (float): Confidence threshold to filter detections.
            a_nms_thre (float): Intersection-over-Union (IoU) threshold for NMS.
            a_top_k_thre (int): Maximum number of top predictions to keep per image.
            a_proc_mode (Literal["batch", "online"]): Processing mode.
                - "batch": Returns all predictions as a nested list.
                - "online": Returns predictions for the first image only.
            a_classes (Optional[Tuple[int, ...]], optional): Filter predictions by class IDs. Defaults to None.
            a_num_classes (Optional[int], optional): Total number of classes in the model. Defaults to None.
            a_multi_label (bool, optional): If True, allows multiple labels per bounding box. Defaults to False.

        Returns:
            IntBBox2DList | IntBBox2DNestedList:
                Structured bounding boxes after scaling and NMS. The return type
                depends on the processing mode.

        Raises:
            RuntimeError: If postprocessing fails, e.g., due to invalid prediction shapes.
        """

        def clip_boxes(
            a_boxes: torch.Tensor | npt.NDArray[Any], a_shape: Tuple[int, int] | npt.NDArray[Any]
        ) -> torch.Tensor | npt.NDArray[Any]:
            h, w = a_shape[:2]
            if isinstance(a_boxes, torch.Tensor):
                a_boxes[..., 0].clamp_(0, w)
                a_boxes[..., 1].clamp_(0, h)
                a_boxes[..., 2].clamp_(0, w)
                a_boxes[..., 3].clamp_(0, h)
            else:
                a_boxes[..., [0, 2]] = a_boxes[..., [0, 2]].clip(0, w)
                a_boxes[..., [1, 3]] = a_boxes[..., [1, 3]].clip(0, h)
            return a_boxes

        def scale_boxes(
            a_boxes: torch.Tensor,
            a_model_size: Tuple[int, int] | npt.NDArray[Any],
            a_data_size: Tuple[int, int] | npt.NDArray[Any],
            a_ratio_pad: Optional[Tuple[int, int]] = None,
            a_pad: bool = True,
            a_is_xywh: bool = False,
        ) -> torch.Tensor | npt.NDArray[Any]:
            if a_ratio_pad is None:
                gain = min(a_model_size[0] / a_data_size[0], a_model_size[1] / a_data_size[1])
                pad_x = round((a_model_size[1] - a_data_size[1] * gain) / 2 - 0.1)
                pad_y = round((a_model_size[0] - a_data_size[0] * gain) / 2 - 0.1)
            else:
                gain = a_ratio_pad[0][0]
                pad_x, pad_y = a_ratio_pad[1]

            if a_pad:
                a_boxes[..., 0] -= pad_x
                a_boxes[..., 1] -= pad_y
                if not a_is_xywh:
                    a_boxes[..., 2] -= pad_x
                    a_boxes[..., 3] -= pad_y
            a_boxes[..., :4] /= gain
            return clip_boxes(a_boxes, a_data_size)

        try:
            if a_backend == "ultralytics":
                assert isinstance(a_preds, list) and all(
                    isinstance(res, UltraResults) for res in a_preds
                ), "For `ultralytics` backend, `a_preds` must be a list of `ultralytics.engine.results.Results`."

                boxes = IntBBox2DNestedList()

                for preds in a_preds:
                    pred_boxes = preds.boxes
                    xyxy = np.asarray(pred_boxes.xyxy.cpu().numpy(), dtype=np.float32)
                    conf = np.asarray(pred_boxes.conf.cpu().numpy(), dtype=np.float32)
                    b_cls = np.asarray(pred_boxes.cls.cpu().numpy(), dtype=np.float32)
                    detections = np.hstack((xyxy, conf[:, None], b_cls[:, None]))
                    boxes_list: IntBBox2DList = IntBBox2DList.from_xyxy(a_coords=detections, a_use_float=False)
                    boxes.append(boxes_list)

            else:
                assert isinstance(a_preds, np.ndarray), "`a_preds` must be a numpy ndarray."
                assert a_data_sizes is not None, "`a_data_sizes` must be provided."
                assert a_model_size is not None, "`a_model_size` must be provided."
                assert a_conf_thre is not None, "`a_conf_thre` must be provided."
                assert a_nms_thre is not None, "`a_nms_thre` must be provided."
                assert a_top_k_thre is not None, "`a_top_k_thre` must be provided."
                assert a_num_classes is not None, "`a_num_classes` must be provided."

                boxes = IntBBox2DNestedList()
                batch_preds = UltralyticsNMS.nms(
                    a_preds=a_preds,
                    a_conf_thre=a_conf_thre,
                    a_iou_thre=a_nms_thre,
                    a_top_k_thre=a_top_k_thre,
                    a_classes=a_classes,
                    a_agnostic=False,
                    a_multi_label=a_multi_label,
                    a_labels=None,
                    a_num_classes=a_num_classes,
                    a_pre_nms_top_k=30000,
                    a_max_wh=7680,
                    a_rotated=False,
                    a_use_nms=True,
                    a_torch_nms_mode="original",
                    a_max_time=0.05,
                    a_break_on_timeout=False,
                    a_return_idxs=False,
                )
                for preds, size in zip(cast(List[torch.Tensor], batch_preds), a_data_sizes):
                    preds[:, :4] = cast(
                        torch.Tensor,
                        scale_boxes(a_boxes=preds[:, :4], a_model_size=a_model_size.to_tuple(), a_data_size=size),
                    )
                    boxes_list = IntBBox2DList.from_xyxy(a_coords=preds.cpu().numpy(), a_use_float=False)
                    boxes.append(boxes_list)

            if a_proc_mode == "online":
                return boxes[0]
            return boxes
        except Exception as e:
            raise RuntimeError(f"Postprocessing failed in {cls.__name__}. Original error: {e}") from e

    @classmethod
    def create_infer_request(
        cls,
        a_images: npt.NDArray[np.float32],
        a_model_name: str,
        a_model_in_layers: Tuple[str],
        a_model_version: Optional[int] = None,
    ) -> predict_pb2.PredictRequest:
        """
        Create a TensorFlow Serving gRPC inference request.

        This method prepares a `PredictRequest` for sending input images to
        a TensorFlow or OVMS model over gRPC. It converts the images into
        a tensor proto and sets the model name and optional version.

        Args:
            a_images (np.ndarray): Preprocessed input images of shape
                `(batch_size, channels, height, width)` or `(batch_size, height, width, channels)`.
            a_model_name (str): Name of the target model to query.
            a_model_in_layers (Tuple[str]): Names of the model input layers.
            a_model_version (Optional[int], optional): Specific version of the model.
                If None, the latest version is used. Defaults to None.

        Returns:
            predict_pb2.PredictRequest: gRPC request object ready for inference.

        Warnings:
            If the request cannot be created, a warning is issued instead of
            raising an exception. This can happen due to invalid input shapes
            or other unexpected errors.
        """
        try:
            request = predict_pb2.PredictRequest()
            request.model_spec.name = a_model_name
            if a_model_version is not None:
                request.model_spec.version.value = a_model_version
            request.inputs[a_model_in_layers[0]].CopyFrom(make_tensor_proto(a_images, shape=None))
            return request
        except Exception as e:
            msg = (
                f"Create Inference Request failed for model='{a_model_name}', "
                f"version='{a_model_version}', input shape={getattr(a_images, 'shape', None)}. "
                f"Exception: {type(e).__name__}: {e}"
            )
            warnings.warn(msg)

    @classmethod
    def decode_preds(
        cls, a_response: predict_pb2.PredictResponse, a_model_out_layers: Tuple[str]
    ) -> Tuple[bool, Dict[str, npt.NDArray[Any]]]:
        """
        Decode predictions from a gRPC `PredictResponse` into NumPy arrays.

        This method extracts tensor outputs from the response for the specified
        output layers, converts them into NumPy arrays, and checks their validity.

        Args:
            a_response (predict_pb2.PredictResponse): The gRPC response returned by
                the model server after an inference request.
            a_model_out_layers (Tuple[str]): Names of the output layers to decode.

        Returns:
            Tuple[bool, Dict[str, np.ndarray]]:
                - A boolean indicating whether all requested outputs were successfully decoded.
                - A dictionary mapping layer names to their corresponding NumPy arrays.
                If decoding fails, the dictionary may be empty or incomplete.

        Warnings:
            - Emits warnings if any output layer is missing, empty, or cannot be converted
            to a NumPy array.
            - Returns `valid=False` if any decoding issue occurs, but does not raise an exception.
        """
        try:
            preds: Dict[str, npt.NDArray[Any]] = {}
            valid: bool = True
            for layer in a_model_out_layers:
                if layer not in a_response.outputs:
                    valid = False
                    warnings.warn(f"Decode Predictions failed: '{layer}' not found in outputs.")
                    break
                tensor_proto = a_response.outputs[layer]
                if not tensor_proto.tensor_content:
                    valid = False
                    warnings.warn(f"Decode Predictions failed: '{layer}' tensor content is empty.")
                    break
                try:
                    array = make_ndarray(tensor_proto)
                except ValueError:
                    valid = False
                    warnings.warn(f"Decode Predictions failed for layer '{layer}': {ve}")
                    break
                if array.ndim == 0 or array.size == 0:
                    valid = False
                    warnings.warn(f"Decode Predictions failed: layer '{layer}' array is empty or has zero dimension.")
                    break
                preds[layer] = array
            return valid, preds
        except Exception as e:
            warnings.warn(f"Decode Predictions encountered an unexpected error: {type(e).__name__}: {e}")
            return False, {}

    def infer(
        self,
        a_images: Union[
            Image2D,
            Frame2D,
            npt.NDArray[np.uint8],
            Sequence[Image2D],
            Sequence[Frame2D],
            Sequence[npt.NDArray[np.uint8]],
        ],
    ) -> IntBBox2DList | IntBBox2DNestedList:
        """
        Perform synchronous object detection inference on input images using the configured model backend.

        The method supports both OpenVINO and OVMS backends. Input images are preprocessed,
        passed through the model, and post-processed to obtain bounding box predictions.

        Args:
            a_images (Union[Image2D, Frame2D, np.ndarray, Sequence[Image2D | Frame2D | np.ndarray]]):
                A single image or a sequence of images/frames to perform inference on.
                Images can be instances of `Image2D` or `Frame2D`, or raw numpy arrays
                of dtype `np.uint8` with shape (H, W, C).

        Returns:
            IntBBox2DList | IntBBox2DNestedList:
                - `IntBBox2DList` if `proc_mode` is "online", containing bounding boxes for a single image.
                - `IntBBox2DNestedList` if `proc_mode` is "batch", containing bounding boxes for each image in the batch.

        Raises:
            RuntimeError: If inference fails due to invalid model configuration,
                preprocessing/postprocessing errors, or backend issues.
            NotImplementedError: If the configured backend is not supported.

        Notes:
            - For the OpenVINO backend, the model must be loaded and compiled.
            - For the OVMS backend, gRPC service stub must be initialized.
            - OVMS inference retries `infer_trial` times if the response is invalid.
        """
        try:
            assert self.model_size is not None, "Model size must be set."
            assert self.conf_thre is not None, "Confidence threshold must be set."
            assert self.nms_thre is not None, "NMS threshold must be set."
            assert self.top_k_thre is not None, "Top-K threshold must be set."
            assert self.multi_label is not None, "Multi-label flag must be set."

            if self.backend == "openvino":
                assert self.model is not None, "Model is not loaded."
                assert isinstance(self.model, ov.CompiledModel), "Invalid OpenVINO model."

                pp_images, data_sizes = YOLO.preproc(
                    a_images=a_images, a_model_size=self.model_size, a_backend=self.backend, a_precision=self._precision
                )
                preds = self.model(pp_images)
                return YOLO.postproc(
                    a_preds=cast(npt.NDArray[Any], preds),
                    a_proc_mode=self._proc_mode,
                    a_backend=self.backend,
                    a_model_size=self.model_size,
                    a_data_sizes=data_sizes,
                    a_conf_thre=self.conf_thre,
                    a_nms_thre=self.nms_thre,
                    a_top_k_thre=self.top_k_thre,
                    a_classes=self.classes,
                    a_num_classes=self.num_classes,
                    a_multi_label=self.multi_label,
                )
            if self.backend == "ovms":
                assert self.model_in_layers is not None, "Model input layers must be set."
                assert self.model_out_layers is not None, "Model output layers must be set."
                assert self.service_stub is not None, "gRPC service stub is not initialized."

                pp_images, data_sizes = YOLO.preproc(
                    a_images=a_images, a_model_size=self.model_size, a_backend=self.backend, a_precision=self._precision
                )
                for tid in range(self.infer_trial):
                    request = YOLO.create_infer_request(
                        a_images=pp_images,
                        a_model_name=self.name,
                        a_model_in_layers=self.model_in_layers,
                        a_model_version=self.id,
                    )
                    response = self.service_stub.Predict(request, timeout=self.infer_timeout)
                    valid, preds = YOLO.decode_preds(a_response=response, a_model_out_layers=self.model_out_layers)
                    if valid:
                        return YOLO.postproc(
                            a_preds=preds[self.model_out_layers[0]],
                            a_proc_mode=self._proc_mode,
                            a_backend=self.backend,
                            a_model_size=self.model_size,
                            a_data_sizes=data_sizes,
                            a_conf_thre=self.conf_thre,
                            a_nms_thre=self.nms_thre,
                            a_top_k_thre=self.top_k_thre,
                            a_classes=self.classes,
                            a_num_classes=self.num_classes,
                            a_multi_label=self.multi_label,
                        )
                    warnings.warn(
                        f"Invalid inference response from OVMS in trial {tid}. "
                        f"Request: model='{self.name}', version='{self.id}', "
                        f"pre_images shape={getattr(pp_images, 'shape', None)}, "
                        f"Response: {response}, Decoded preds: {preds}"
                    )
                return IntBBox2DList()
            if self.backend == "ultralytics":
                assert isinstance(self.model, ultra.YOLO), "Invalid Ultralytics YOLO model."
                pp_images = YOLO.preproc(
                    a_images=a_images, a_model_size=self.model_size, a_backend=self.backend, a_precision=self._precision
                )
                preds = self.model.predict(
                    source=pp_images,
                    conf=self._conf_thre,
                    iou=self._nms_thre,
                    max_det=self._top_k_thre,
                    classes=self._classes,
                )
                return YOLO.postproc(
                    a_preds=cast(npt.NDArray[Any], preds),
                    a_proc_mode=self._proc_mode,
                    a_backend=self.backend,
                )
            raise NotImplementedError(
                f"Inference for backend `{self.backend}` is not implemented. "
                f"Support backends: `openvino`, `ovms`, `ultralytics`."
            )
        except Exception as e:
            raise RuntimeError(f"Inference failed in `{self.name}`. Original error: {e}") from e

    async def infer_async(self) -> None:
        """
        Perform asynchronous object detection inference using a queue-based IO interface.

        This method continuously fetches frames from the input queue, preprocesses them,
        performs inference using OVMS backend via gRPC, and posts the detection results
        back to the output queue. Supports multiple inference trials in case of invalid responses.

        The method is intended to be run as a background task and requires an active stop event
        to gracefully terminate the loop.

        Raises:
            AssertionError: If required attributes (stop_event, io, executor, model configuration,
                or gRPC stub) are not set.
            ValueError: If all inference trials fail for a given frame.
            RuntimeError: If asynchronous inference fails due to unexpected errors.
            NotImplementedError: If the backend, IO mode, or concurrency mode is unsupported.

        Notes:
            - Supported configuration: backend='ovms', io_mode='queue', conc_mode must be set.
            - Frames are processed asynchronously using `asyncio` and offloaded to threads for
            CPU-bound preprocessing and postprocessing.
            - The number of inference retries is controlled by `self.infer_trial`.
        """
        assert self.stop_event is not None, "Stop event must be set."

        try:
            assert self.io is not None, "IO interface must be set."
            assert self.executor is not None, "Executor must be set."
            assert self.model_size is not None, "Model size must be set."
            assert self.top_k_thre is not None, "Top-K threshold must be set."
            assert self.conf_thre is not None, "Confidence threshold must be set."
            assert self.nms_thre is not None, "NMS threshold must be set."
            assert self.model_in_layers is not None, "Model input layers must be set."
            assert self.model_out_layers is not None, "Model output layers must be set."
            assert self.service_stub is not None, "gRPC service stub is not initialized."

            if self.backend == "ovms" and self.io_mode == "queue" and self.conc_mode is not None:
                while not self.stop_event.is_set() and not self.io.is_input_done():
                    frame = await self.io.get_input_async()
                    if frame is None:
                        break
                    pp_images, data_sizes = await asyncio.to_thread(
                        YOLO.preproc, frame, self.model_size, self.backend, self._precision
                    )
                    for tid in range(self.infer_trial):
                        request = await asyncio.to_thread(
                            YOLO.create_infer_request,
                            pp_images,
                            self.name,
                            self.model_in_layers,
                            self.model_version,
                        )
                        response = await self.service_stub.Predict(request, timeout=self.infer_timeout)
                        valid, preds = await asyncio.to_thread(YOLO.decode_preds, response, self.model_out_layers)
                        if valid:
                            break
                        else:
                            warnings.warn(
                                f"Invalid inference response from OVMS in trial {tid}, "
                                f"response: {response}, preds: {preds}, trial: {tid}."
                            )
                    if valid:
                        boxes = await asyncio.to_thread(
                            YOLO.postproc,
                            preds[self.model_out_layers[0]],
                            self._proc_mode,
                            self.backend,
                            data_sizes,
                            self.model_size,
                            self.conf_thre,
                            self.nms_thre,
                            self.top_k_thre,
                            self.classes,
                            self.num_classes,
                            self.multi_label,
                        )
                    else:
                        boxes = IntBBox2DList()
                        raise ValueError(
                            f"All inference trials failed for frame {frame}. " f"Number of trials: {self.infer_trial}."
                        )
                    await self.io.put_output_async((frame, boxes))

            else:
                raise NotImplementedError(
                    f"Async inference is not implemented for the given configuration: "
                    f"backend='{self.backend}', io_mode='{self.io_mode}', conc_mode='{self.conc_mode}'. "
                    f"Supported configuration: backend='ovms', io_mode='queue', conc_mode set."
                )
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Asynchronous inference failed in `{self.name}`. Original error: {e}") from e

    async def run_async(self) -> None:
        """
        Launch multiple asynchronous inference workers for processing input frames.

        This method manages a pool of asynchronous tasks that run `infer_async` concurrently
        to perform object detection on frames fetched from a queue-based IO interface.
        It ensures all workers stop gracefully when the stop event is triggered and
        signals when output processing is complete.

        Raises:
            AssertionError: If required attributes (stop_event, max_workers, io) are not set or invalid.
            RuntimeError: If an unexpected error occurs during asynchronous execution.
            NotImplementedError: If the current combination of call_mode, backend, and io_mode
                is unsupported. Currently, only:
                    call_mode='async', backend='ovms', io_mode='queue' is supported.

        Notes:
            - Each worker is an independent asynchronous task that repeatedly calls `infer_async`.
            - The number of concurrent workers is controlled by `self.max_workers`.
            - After all workers complete, `io.output_done_async` is called to signal completion.
        """
        assert self.stop_event is not None, "Stop event must be set."

        try:
            if self.call_mode == 'async' and self.backend == "ovms" and self.io_mode == "queue":
                assert (
                    self.max_workers is not None and self.max_workers > 0
                ), "Max workers must be set and greater than 0."
                assert self.io is not None, "IO must be set."

                try:
                    async with asyncio.TaskGroup() as tg:
                        for wid in range(self.max_workers):
                            tg.create_task(self.infer_async(), name=f"worker-{wid}")
                finally:
                    await self.io.output_done_async()
            else:
                raise NotImplementedError(
                    f"This combination of parameters is not yet implemented: "
                    f"call_mode='{self.call_mode}', backend='{self.backend}', io_mode='{self.io_mode}'. "
                    f"Only call_mode='async', backend='ovms', io_mode='queue' is currently supported."
                )
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Async Run failed: {e}") from e
