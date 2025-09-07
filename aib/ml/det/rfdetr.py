"""Machine Learning - Object Detection - RF-DETR Model

This module implements the RFDETR class, which provides synchronous and asynchronous object detection using
the RF-DETR (Relation-Free Detection Transformer) architecture. The detector supports multiple backends,
including OpenVINO (for local inference) and OVMS (OpenVINO Model Server) for remote inference via gRPC.
It includes utilities for image preprocessing, model inference, and postprocessing to structured bounding boxes.

Classes:
    RFDETR: Main detection class for general object detection tasks.

Type Variables:
    IOT: TypeVar for input/output interface, typically a QueueIO handling
         frames and bounding box outputs.
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
from tensorflow import make_ndarray, make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from aib.cnt.io import QueueIO
from aib.cv.geom.box.bbox2d import IntBBox2DList, IntBBox2DNestedList
from aib.cv.geom.size import IntSize
from aib.cv.img.frame import Frame2D
from aib.cv.img.image import Image2D
from aib.misc.common_types import StopEvent
from aib.ml.det.utils.b_det_mdl import BaseDetModel

IOT = TypeVar(
    "IOT",
    bound=Union[
        QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DList]],
        QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DNestedList]],
    ],
    default=QueueIO[None | Frame2D, None | Tuple[Frame2D, IntBBox2DList]],
)


class RFDETR(BaseDetModel[IOT]):
    """RF-DETR Object Detector.

    Implements an object detection model based on the Relation-Free Detection Transformer (RF-DETR)
    architecture. Supports both OpenVINO and OVMS backends for synchronous and asynchronous inference.

    The class provides preprocessing and postprocessing utilities, including resizing, normalization,
    bounding box extraction, and filtering based on confidence, NMS, and top-K thresholds.

    Features:
        - Backend support: `openvino`, `ovms`.
        - Inference modes: synchronous (`infer`) and asynchronous (`infer_async` / `run_async`).
        - Input/Output handling via `QueueIO` or direct arguments.
        - Profiling support with `Profiler`.
        - Flexible configuration for model size, classes, device, precision, and concurrency.

    Attributes:
        id (Optional[int]): Unique identifier for the object instance.
        name (str): Human-readable name of the object (default: 'RFDETR').
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
        a_model_size: Optional[IntSize] = IntSize(width=560, height=560),
        a_model_config: Optional[Dict[str, Any]] = None,
        a_model_in_layers: Optional[Tuple[str, ...]] = ("input",),
        a_model_out_layers: Optional[Tuple[str, ...]] = ("labels", "dets"),
        a_backend_core: Optional[ov.Core | Any] = None,
        a_data_size: Optional[IntSize] = None,
        a_infer_timeout: Optional[float] = None,
        a_infer_trial: int = 1,
        a_device: Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"] = "AUTO",
        a_precision: Literal["FP32", "FP16", "INT8"] = "FP32",
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys"] = "ovms",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'RFDETR',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the RFDETR class with configurable parameters for model inference and processing.

        Args:
            a_conf_thre (Optional[float]): Confidence threshold for detections.
            a_nms_thre (Optional[float]): Non-maximum suppression threshold.
            a_top_k_thre (Optional[int]): Top-K threshold for filtering detections.
            a_min_size_thre (Optional[IntSize]): Minimum size threshold for detected objects.
            a_classes (Optional[Tuple[int, ...]]): List of class indices to detect.
            a_num_classes (Optional[int], default=None):
                Total number of classes the model can predict.
            a_multi_label (bool, default=False):
                Indicates if multiple labels can be assigned to a single sample.
            a_backend_core (Optional[ov.Core]): OpenVINO core object for inference.
            a_model_uri (Optional[str | bytes | object | PathLike[str]]): URI or path to the model.
            a_model_version (Optional[int]): Version of the model to use.
            a_model_size (Optional[IntSize]): Input size for the model (default: IntSize(width=560, height=560)).
            a_model_config (Optional[Dict[str, Any]]): Additional model configuration parameters.
            a_data_size (Optional[IntSize]): Size of the input data.
            a_infer_timeout (Optional[float]): Timeout for inference in seconds.
            a_infer_trial (int): Number of inference trials (default: 1).
            a_device (Literal["CPU", "GPU", "MYRIAD", "FPGA", "HETERO", "AUTO"]):
                Device to run inference on (default: "AUTO").
            a_precision (Literal["FP32", "FP16", "INT8"], optional):
                Model numerical precision. Defines the data type for weights, inputs, and outputs. Defaults to "FP32".
            a_call_mode (Literal["sync", "async"]): Inference call mode (default: 'sync').
            a_io_mode (Literal["args", "queue", "ipc"]): Input/output mode (default: "args").
            a_proc_mode (Literal["batch", "online"]): Processing mode (default: "online").
            a_backend (Literal["ovms", "openvino", "sys", "opencv"]): Backend for inference (default: "ovms").
            a_conc_mode (Optional[Literal["thread", "process"]]): Concurrency mode.
            a_max_workers (Optional[int]): Maximum number of worker threads or processes.
            a_io (Optional[IOT]): IO handler object.
            a_stop_event (Optional[StopEvent]): Event to signal stopping of processing.
            a_id (Optional[int]): Unique identifier for the detector instance.
            a_name (str): Name of the detector (default: 'RFDETR').
            a_use_prof (bool): Enable profiling (default: False).
            a_use_cfg (bool): Enable configuration usage (default: True).
            a_use_log (bool): Enable logging (default: True).
            **kwargs (Any): Additional keyword arguments for further customization.
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
        """Load the detection model for inference.

        This method initializes the model depending on the configured backend:
        - For `openvino`, it reads and compiles the model using the OpenVINO Runtime Core.
        - For `ovms` (OpenVINO Model Server), it sets up a gRPC channel and service stub
          for synchronous or asynchronous inference requests.

        The method also infers the input model size if not already set.

        Raises:
            AssertionError: If required attributes such as `model_uri`, `core`, or `device` are missing.
            NotImplementedError: If the backend is unsupported.
            RuntimeError: If the model fails to load for any other reason.
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

            else:
                raise NotImplementedError(
                    f"Loading models for backend `{self.backend}` is not implemented. "
                    f"Supported backends: `openvino`, `ovms`."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from: {self.model_uri}. Original error: {e}") from e

    @classmethod
    def preproc(
        cls,
        a_images: Union[
            Image2D,
            Frame2D,
            npt.NDArray[np.uint8],
            Sequence[Image2D],
            Sequence[Frame2D],
            Sequence[npt.NDArray[np.uint8]],
        ],
        a_model_size: IntSize,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Preprocess input images for the detection model.

        This method converts input images to the format expected by the model,
        including resizing to the model's input size, normalizing pixel values,
        and converting from HWC to CHW format. Both single images and sequences
        of images are supported.

        Args:
            a_images: Single image or a sequence of images. Supported types include:
                - `Image2D` or `Frame2D` instances
                - NumPy arrays of shape (H, W, C) with dtype `uint8`
                - Sequences of any of the above
            a_model_size: Target input size for the model, specified as `IntSize(width, height)`.

        Returns:
            Tuple containing:
            - `np.ndarray` of preprocessed image data with shape (batch, 3, height, width) and dtype `float32`
            - `np.ndarray` of original image sizes with shape (batch, 2) and dtype `float32`,
                where each entry is [height, width]

        Raises:
            RuntimeError: If preprocessing fails due to invalid input types or shapes.
        """

        def pre_transform(
            a_image: npt.NDArray[np.uint8], a_target_height: int, a_target_width: int
        ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(a_image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w, _ = a_image.shape

            # Resize to model input size
            img_resized = cv2.resize(img_rgb, (a_target_width, a_target_height)).astype(np.float32) / 255.0

            # Normalize using ImageNet mean/std
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_resized = (img_resized - means) / stds

            # HWC -> CHW
            img_resized = np.transpose(img_resized, (2, 0, 1))

            # Add batch dimension
            img_resized = np.expand_dims(img_resized, axis=0)

            # Batch size
            data_size = np.array([[orig_h, orig_w]], dtype=np.float32)
            return img_resized, data_size

        try:
            if isinstance(a_images, (list, tuple)):
                images: List[npt.NDArray[np.float32]] = []
                data_sizes: List[npt.NDArray[np.float32]] = []
                for image in a_images:
                    if isinstance(image, (Image2D, Frame2D)):
                        image = image.data
                    image, size = pre_transform(image, a_model_size.height, a_model_size.width)
                    images.append(image)
                    data_sizes.append(size)
            else:

                if isinstance(a_images, (Image2D, Frame2D)):
                    image = a_images.data
                else:
                    image = a_images
                image, size = pre_transform(cast(npt.NDArray[np.uint8], image), a_model_size.height, a_model_size.width)
                images = [image]
                data_sizes = [size]

            return np.concatenate(images, axis=0), np.concatenate(data_sizes, axis=0)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed in {cls.__name__}. Original error: {e}") from e

    @classmethod
    def postproc(
        cls,
        a_dets: npt.NDArray[np.float32],
        a_logits: npt.NDArray[np.float32],
        a_data_sizes: npt.NDArray[np.float32],
        a_top_k_thre: int,
        a_conf_thre: float,
        a_proc_mode: Literal["batch", "online"],
    ) -> IntBBox2DList | IntBBox2DNestedList:
        """Post-process raw detection outputs into bounding box lists.

        This method converts raw model outputs (`a_dets` and `a_logits`) into
        structured bounding box objects, applying confidence thresholding,
        top-K selection, and coordinate scaling from normalized to absolute
        image dimensions.

        Args:
            a_dets (np.ndarray): Raw detection coordinates from the model of shape [batch, num_boxes, 4].
            a_logits (np.ndarray): Raw class logits from the model of shape [batch, num_boxes, num_classes].
            a_data_sizes (np.ndarray): Original image sizes as a float array of shape [batch, 2] (height, width).
            a_top_k_thre (int): Maximum number of top-scoring detections to keep per image.
            a_conf_thre (float): Confidence threshold; detections below this are discarded.
            a_proc_mode (str):
                Processing mode; `"online"` returns only the first image's detections,
                otherwise returns detections for all images.

        Returns:
            IntBBox2DList | IntBBox2DNestedList: Structured bounding boxes for each image in the batch.

        Raises:
            RuntimeError: If post-processing fails due to shape mismatches or invalid inputs.
        """

        def cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
            x_c, y_c, w, h = x.unbind(-1)
            b = [
                (x_c - 0.5 * w.clamp(min=0.0)),
                (y_c - 0.5 * h.clamp(min=0.0)),
                (x_c + 0.5 * w.clamp(min=0.0)),
                (y_c + 0.5 * h.clamp(min=0.0)),
            ]
            return torch.stack(b, dim=-1)

        try:
            boxes = IntBBox2DNestedList()

            dets: torch.Tensor = torch.from_numpy(a_dets)
            logits: torch.Tensor = torch.from_numpy(a_logits)
            data_sizes: torch.Tensor = torch.from_numpy(a_data_sizes)

            assert len(logits) == len(data_sizes)
            assert data_sizes.shape[1] == 2

            prob = logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), a_top_k_thre, dim=1)
            batch_scores = topk_values
            topk_boxes = topk_indexes // logits.shape[2]
            batch_labels = topk_indexes % logits.shape[2]
            batch_boxes = cxcywh_to_xyxy(dets)
            batch_boxes = torch.gather(batch_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = data_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            batch_boxes = batch_boxes * scale_fct[:, None, :]

            for scores, labels, coords in zip(batch_scores, batch_labels, batch_boxes):
                keep = scores > a_conf_thre
                scores = scores[keep]
                labels = labels[keep]
                coords = coords[keep]

                boxes_list = IntBBox2DList.from_xyxy(
                    a_coords=coords.float().cpu().numpy(),
                    a_scores=scores.float().cpu().numpy(),
                    a_labels=labels.int().cpu().numpy(),
                    a_use_float=False,
                )
                boxes.append(boxes_list)

            if a_proc_mode == "online":
                return boxes[0]
            return boxes
        except Exception as e:
            msg = f"Postprocessing failed: {e}"
            raise RuntimeError(msg) from e

    @classmethod
    def create_infer_request(
        cls,
        a_images: npt.NDArray[np.float32],
        a_model_name: str,
        a_model_in_layers: Tuple[str],
        a_model_version: Optional[int] = None,
    ) -> predict_pb2.PredictRequest:
        """Create a gRPC inference request for TensorFlow Serving (OVMS).

        Constructs a `PredictRequest` object containing the input images
        and model specifications to be sent to the OVMS server.

        Args:
            a_images:
                NumPy array of preprocessed images, typically of shape (batch_size, channels, height, width) and
                dtype `float32`.
            a_model_name: Name of the model to query on the OVMS server.
            a_model_in_layers (Tuple[str]):
                Tuple of input layer names expected by the model. The first
                entry is used to populate the request with image data.
            a_model_version: Optional version of the model. If `None`, the latest version is used.

        Returns:
            predict_pb2.PredictRequest: The constructed inference request ready
            to be sent via gRPC.

        Raises:
            Warnings: If an exception occurs during request creation, a warning
                is emitted with information about the model, version, and input
                shape. The method does not raise an exception but issues a warning.
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
        """Decode predictions from a TensorFlow Serving (OVMS) response.

        Converts the raw `PredictResponse` outputs into NumPy arrays and
        validates the presence and integrity of required output layers.

        Args:
            a_response: `PredictResponse` object returned by the OVMS server
                after an inference request.
            a_model_out_layers (Tuple[str]):
                Tuple of expected output layer names (e.g., `("labels", "dets")`).
                Each layer will be retrieved and converted into a NumPy array.

        Returns:
            A tuple containing:
                - valid (bool):
                    True if the response contains valid and non-empty outputs for all required layers, False otherwise.
                - preds (dict):
                    Dictionary mapping output layer names (`labels`, `dets`) to their corresponding NumPy arrays.
                    If `valid` is False, this may be an empty dictionary.

        Warnings:
            Emits warnings in case:
                - Required layers are missing in the response.
                - Tensor content is empty.
                - Conversion from TensorProto to ndarray fails.
                - Output arrays are empty or have zero dimensions.
                - Any unexpected error occurs during decoding.
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
        """Perform synchronous inference on input images.

        This method preprocesses input images, executes the model inference
        on the selected backend (`openvino` or `ovms`), and postprocesses
        the predictions to return bounding boxes. Supports both single images
        and batches of images.

        Args:
            a_images: Single image or a sequence of images. Supported types include:
                - `Image2D` or `Frame2D` instances
                - NumPy arrays of shape (H, W, C) with dtype `uint8`
                - Sequences of any of the above

        Returns:
            - `IntBBox2DList` if `proc_mode` is "online" or for single-image inference
            - `IntBBox2DNestedList` for batch inference when `proc_mode` is "batch"

        Raises:
            AssertionError: If essential parameters such as `model_size`, `top_k_thre`,
                or `conf_thre` are not set.
            NotImplementedError: If the specified backend or configuration is unsupported.
            RuntimeError: If inference fails due to preprocessing, model execution,
                or postprocessing errors.
        """
        try:
            assert self.model_size is not None, "Model size must be set."
            assert self.top_k_thre is not None, "Top-K threshold must be set."
            assert self.conf_thre is not None, "Confidence threshold must be set."

            if self.backend == "openvino":
                assert self.model is not None, "Model is not loaded."

                pp_images, data_sizes = RFDETR.preproc(a_images=a_images, a_model_size=self.model_size)
                preds = self.model(pp_images)
                dets = cast(npt.NDArray[np.float32], preds[self.model.output(0)])
                logits = cast(npt.NDArray[np.float32], preds[self.model.output(1)])
                return RFDETR.postproc(
                    a_dets=dets,
                    a_logits=logits,
                    a_data_sizes=data_sizes,
                    a_top_k_thre=self.top_k_thre,
                    a_conf_thre=self.conf_thre,
                    a_proc_mode=self.proc_mode,
                )

            if self.backend == "ovms":
                assert self.model_in_layers is not None, "Model input layers must be set."
                assert self.model_out_layers is not None, "Model output layers must be set."
                assert self.service_stub is not None, "gRPC service stub is not initialized."

                pp_images, data_sizes = RFDETR.preproc(a_images=a_images, a_model_size=self.model_size)
                for tid in range(self.infer_trial):
                    request = RFDETR.create_infer_request(
                        a_images=pp_images,
                        a_model_name=self.name,
                        a_model_in_layers=self.model_in_layers,
                        a_model_version=self.id,
                    )
                    response = self.service_stub.Predict(request, timeout=self.infer_timeout)
                    valid, preds = RFDETR.decode_preds(a_response=response, a_model_out_layers=self.model_out_layers)
                    if valid:
                        return RFDETR.postproc(
                            a_dets=preds['dets'],
                            a_logits=preds['labels'],
                            a_data_sizes=data_sizes,
                            a_top_k_thre=self.top_k_thre,
                            a_conf_thre=self.conf_thre,
                            a_proc_mode=self.proc_mode,
                        )
                    warnings.warn(
                        f"Invalid inference response from OVMS in trial {tid}. "
                        f"Request: model='{self.name}', version='{self.id}', "
                        f"pre_images shape={getattr(pp_images, 'shape', None)}, "
                        f"Response: {response}, Decoded preds: {preds}"
                    )
                return IntBBox2DList()

            raise NotImplementedError(
                f"Inference for backend `{self.backend}` is not implemented. " f"Support backends: `openvino`, `ovms`."
            )
        except Exception as e:
            raise RuntimeError(f"Inference failed in `{self.name}`. Original error: {e}") from e

    async def infer_async(self) -> None:
        """Perform asynchronous inference on frames from the input queue.

        This method continuously reads frames from the input queue, preprocesses them,
        sends inference requests to an OVMS server, decodes the predictions, post-processes
        them into bounding boxes, and writes the results to the output queue. It stops
        processing when the `stop_event` is set or when the input queue is exhausted.

        Raises:
            AssertionError: If required attributes (`stop_event`, `io`, `executor`,
                `model_size`, `top_k_thre`, `conf_thre`) are not set.
            ValueError: If all inference trials fail for a frame.
            NotImplementedError: If called with unsupported backend, io_mode, or concurrency
                configuration.
            RuntimeError: If any unexpected error occurs during async inference. The
                `stop_event` is set before raising this exception.
        """
        assert self.stop_event is not None, "Stop event must be set."

        try:
            assert self.io is not None, "IO interface must be set."
            assert self.executor is not None, "Executor must be set."
            assert self.model_size is not None, "Model size must be set."
            assert self.top_k_thre is not None, "Top-K threshold must be set."
            assert self.conf_thre is not None, "Confidence threshold must be set."
            assert self.model_in_layers is not None, "Model input layers must be set."
            assert self.model_out_layers is not None, "Model output layers must be set."
            assert self.service_stub is not None, "gRPC service stub is not initialized."

            if self.backend == "ovms" and self.io_mode == "queue" and self.conc_mode is not None:
                while not self.stop_event.is_set() and not self.io.is_input_done():
                    frame = await self.io.get_input_async()
                    if frame is None:
                        break
                    pre_images, data_sizes = await asyncio.to_thread(RFDETR.preproc, frame, self.model_size)

                    for tid in range(self.infer_trial):
                        request = await asyncio.to_thread(
                            RFDETR.create_infer_request,
                            pre_images,
                            self.name,
                            self.model_in_layers,
                            self.model_version,
                        )
                        response = await self.service_stub.Predict(request, timeout=self.infer_timeout)
                        valid, preds = await asyncio.to_thread(RFDETR.decode_preds, response, self.model_out_layers)

                        if valid:
                            break
                        else:
                            warnings.warn(
                                f"Invalid inference response from OVMS in trial {tid}, "
                                f"response: {response}, preds: {preds}, trial: {tid}."
                            )

                    if valid:
                        boxes = await asyncio.to_thread(
                            RFDETR.postproc,
                            preds['dets'],
                            preds['labels'],
                            data_sizes,
                            self.top_k_thre,
                            self.conf_thre,
                            self.proc_mode,
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
        """Run asynchronous inference workers to process input frames concurrently.

        This method creates multiple asynchronous workers that call `infer_async`
        to continuously process frames from the input queue and write results
        to the output queue. The number of concurrent workers is controlled by
        `max_workers`. The method ensures that the output queue is properly
        marked as done after all workers finish.

        Raises:
            AssertionError: If required attributes (`stop_event`, `io`, `max_workers`) are not set
                or invalid.
            NotImplementedError: If called with an unsupported combination of
                `call_mode`, `backend`, or `io_mode`.
            RuntimeError: If any unexpected error occurs during asynchronous execution.
                The `stop_event` is set before raising this exception.
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
