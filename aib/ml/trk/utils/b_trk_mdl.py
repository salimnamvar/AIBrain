"""Machine Learning - Object Tracking - Base Abstract Multi-Object Tracking Models

Classes:
    BaseTrkModel: TODO
"""

from abc import ABC
from os import PathLike
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import openvino as ov

from src.utils.cnt.b_dict import BaseDict
from src.utils.cnt.io import BaseIO
from src.utils.cv.geom.box import AnyBox, AnyBoxList
from src.utils.cv.geom.size import IntSize
from src.utils.cv.img.frame import Frame2D
from src.utils.cv.img.image import Image2D
from src.utils.misc.common_types import StopEvent
from src.utils.ml.utils.b_ml_mdl import BaseMLModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseTrkModel(BaseMLModel[IOT], ABC):
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
        return self._src_ids

    @property
    def conf_thre(self) -> Optional[float]:
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
        raise NotImplementedError("Subclasses must implement `infer` method.")
