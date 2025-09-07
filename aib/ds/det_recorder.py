"""Detection Recorder

This module defines the `DetectionRecorder` class, which is responsible for recording detection data.

Classes:
    DetectionRecorder: Responsible for recording detection data and saving it in Parquet format.

Type Variables:
    IOT: Type variable for input/output operations.
"""

import asyncio
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.cnt.b_dict import BaseDict
from src.utils.cnt.io import BaseIO
from src.utils.cv.geom.box.bbox2d import AnyBBox2D, AnyBBox2DList, FloatBBox2D, FloatBBox2DList
from src.utils.cv.img.frame import Frame2D
from src.utils.misc.common_types import StopEvent
from src.utils.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, None | Tuple[Frame2D, AnyBBox2DList | BaseDict[int, AnyBBox2D]]])


class DetectionRecorder(BaseModel[IOT]):
    """Detection Recorder

    Records frame-level metadata and bounding box detections, and writes
    them to a Parquet file with optional compression and metadata.

    The recorder can be run in synchronous or asynchronous mode and is
    designed to integrate with IO pipelines (e.g., queue-based streaming).

    Attributes:
        _frame_columns (Optional[List[str]]): Frame attributes to record. Defaults to all.
        _box_columns (Optional[List[str]]): Box attributes to record. Defaults to standard fields.
        _frame_prefix (str): Prefix for frame attributes in output.
        _box_prefix (str): Prefix for box attributes in output.
        _compression (str): Compression codec (e.g., "snappy", "gzip").
        _extra_metadata (Dict[str, Any]): Metadata to embed in the Parquet schema.
        _records (List[Dict[str, Any]]): Accumulated detection records.
    """

    def __init__(
        self,
        a_frame_columns: Optional[List[str]] = None,
        a_box_columns: Optional[List[str]] = None,
        a_frame_prefix: str = "frame_",
        a_box_prefix: str = "box_",
        a_compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] = "snappy",
        a_extra_metadata: Optional[Dict[str, Any]] = None,
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["sys", "pyarrow"] = "pyarrow",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = 1,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'DetectionRecorder',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Box Detection Pipeline Recorder.

        Args:
            a_frame_columns (Optional[List[str]]):
                List of frame attribute names to record. If None, all available frame attributes will be included.
            a_box_columns (Optional[List[str]]):
                List of bounding box attribute names to record. If None, defaults to
                ["x1", "y1", "x2", "y2", "score", "label"].
            a_frame_prefix (str): Prefix applied to all frame-related columns in the output. Defaults to `"frame_"`.
            a_box_prefix (str): Prefix applied to all bounding-box-related columns in the output. Defaults to `"box_"`.
            a_compression (Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']):
                Compression codec to use for Parquet serialization. Defaults to `"snappy"`.
            a_extra_metadata (Optional[Dict[str, Any]]):
                Extra metadata to embed in the Parquet file schema. Keys and values are stored as strings.
            a_call_mode (Literal["sync", "async"]):
                Execution mode of the recorder. `"sync"` runs in blocking mode, `"async"` integrates with asyncio.
                Defaults to `"sync"`.
            a_io_mode (Literal["args", "queue", "ipc"]):
                IO communication mode.
                    - `"args"` → direct function argument passing.
                    - `"queue"` → queue-based producer/consumer model.
                    - `"ipc"` → inter-process communication.
                Defaults to `"args"`.
            a_proc_mode (Literal["batch", "online"]):
                Processing mode.
                    - `"batch"` → accumulate and process in bulk.
                    - `"online"` → process frame-by-frame.
                Defaults to `"online"`.
            a_backend (Literal["sys", "pyarrow"]): Backend source type for detections. Defaults to `"pyarrow"`.
            a_conc_mode (Optional[Literal["thread", "process"]]):
                Concurrency model for async execution.
                    `"thread"` for multi-threading, `"process"` for multiprocessing.
                    Only relevant in async + queue/ipc modes.
                Defaults to None.
            a_max_workers (Optional[int]):
                Maximum number of worker tasks to spawn in async mode. Must be > 0 if concurrency is enabled.
                Defaults to 1.
            a_io (Optional[IOT]):
                IO handler instance (subclass of `BaseIO`). Required if `a_io_mode` is `"queue"` or `"ipc"`.
            a_stop_event (Optional[StopEvent]):
                Event flag to coordinate stopping across async workers. Must be provided for async execution.
            a_id (Optional[int]):
                Unique identifier for this recorder instance. Defaults to None.
            a_name (str):
                Human-readable name of the recorder instance. Defaults to `"DetectionRecorder"`.
            a_use_prof (bool):
                Enable runtime profiling for performance measurement. Defaults to False.
            a_use_cfg (bool):
                Enable logging of configuration on initialization. Defaults to True.
            a_use_log (bool):
                Enable logging messages during execution. Defaults to True.
            **kwargs (Any):
                Additional keyword arguments passed to the parent `BaseModel`.
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
        self._frame_columns: Optional[List[str]] = a_frame_columns
        self._box_columns: Optional[List[str]] = a_box_columns
        self._frame_prefix: str = a_frame_prefix
        self._box_prefix: str = a_box_prefix
        self._compression: str = a_compression
        self._extra_metadata: Dict[str, Any] = a_extra_metadata or {}
        self._records: List[Dict[str, Any]] = []

    def append(self, a_frame: Frame2D, a_boxes: FloatBBox2DList | BaseDict[int, FloatBBox2D]) -> None:
        """Append detection records for a frame and its bounding boxes.

        For each bounding box, a record containing frame and box attributes
        is added to the recorder’s internal list. If no boxes are provided,
        a single record is created with frame attributes and empty box fields.

        Args:
            a_frame: Frame object containing frame-level metadata.
            a_boxes: Collection of bounding boxes (list or dict).
                - If dict, keys are treated as tracking IDs and added as `box_trk_id`.
                - If empty, an empty-box record is still appended.

        Note:
            Records are only held in memory until `save()` is called.
        """
        is_dict = isinstance(a_boxes, dict) or isinstance(a_boxes, BaseDict)
        items = a_boxes.items() if is_dict else enumerate(a_boxes)
        has_boxes = len(a_boxes) > 0

        for key, bbox in items:
            rec = {}
            frame_fields = self._frame_columns if self._frame_columns is not None else a_frame.__dict__.keys()
            for k in frame_fields:
                rec[f"{self._frame_prefix}{k}"] = getattr(a_frame, k, None)
            if self._box_columns is None:
                rec.update(
                    {
                        f"{self._box_prefix}x1": bbox.p1.x,
                        f"{self._box_prefix}y1": bbox.p1.y,
                        f"{self._box_prefix}x2": bbox.p2.x,
                        f"{self._box_prefix}y2": bbox.p2.y,
                        f"{self._box_prefix}score": bbox.score,
                        f"{self._box_prefix}label": bbox.label,
                    }
                )
            else:
                for k in self._box_columns:
                    rec[f"{self._box_prefix}{k}"] = getattr(bbox, k, None)
            if is_dict:
                rec[f"{self._box_prefix}trk_id"] = key
            self._records.append(rec)

        if not has_boxes:
            rec = {
                f"{self._frame_prefix}{k}": getattr(a_frame, k, None)
                for k in (self._frame_columns or a_frame.__dict__.keys())
            }
            rec.update(
                {
                    f"{self._box_prefix}{k}": None
                    for k in (self._box_columns or ["x1", "y1", "x2", "y2", "score", "label"])
                }
            )
            if is_dict:
                rec[f"{self._box_prefix}trk_id"] = None
            self._records.append(rec)

    def save(
        self,
        a_parquet_path: str | PathLike[str],
        a_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write accumulated records to a Parquet file.

        Args:
            a_parquet_path: Output path for the Parquet file.
            a_metadata: Optional metadata to merge with `_extra_metadata`.

        Behavior:
            - Metadata is converted to string and stored in schema.
            - After saving, the internal record list is cleared.
        """
        parquet_path = Path(a_parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._records)
        table = pa.Table.from_pandas(df)
        metadata = self._extra_metadata
        if a_metadata:
            metadata.update(a_metadata)
        if metadata:
            table = table.replace_schema_metadata({k: str(v) for k, v in metadata.items()})
        pq.write_table(
            table,
            parquet_path,
            compression=self._compression,
            use_deprecated_int96_timestamps=True,
        )

    async def infer_async(self) -> None:
        """Asynchronously consume IO output and append detection records.

        Continuously fetches (frame, boxes) tuples from the IO object until:
            - The stop event is set, or
            - IO signals that output is complete.

        Raises:
            RuntimeError: If any error occurs during async recording.
        """
        assert self.stop_event is not None, "Stop event must be set."
        assert self.io is not None, "IO must be set."

        try:
            if self.call_mode == 'async' and self._io_mode == "queue":
                while not self.stop_event.is_set() and not self.io.is_output_done():
                    out = await self.io.get_output_async()
                    if out is None:
                        break
                    frame, boxes = out
                    await asyncio.to_thread(self.append, frame, boxes)
            else:
                raise NotImplementedError("Configuration not supported.")
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Async Inference failed: {e}") from e

    async def run_async(self) -> None:
        """Run the recorder asynchronously with multiple workers.

        Spawns `max_workers` concurrent tasks, each executing `infer_async()`.
        This is intended for queue-based async pipelines.

        Raises:
            RuntimeError: If async run fails or configuration is invalid.
        """
        assert self.stop_event is not None, "Stop event must be set."

        try:
            if self.call_mode == 'async' and self.io_mode == "queue":
                assert (
                    self.max_workers is not None and self.max_workers > 0
                ), "Max workers must be set and greater than 0."
                assert self.io is not None, "IO must be set."

                async with asyncio.TaskGroup() as tg:
                    for wid in range(self.max_workers):
                        tg.create_task(self.infer_async(), name=f"worker-{wid}")
            else:
                raise NotImplementedError("Configuration not supported.")
        except Exception as e:
            self.stop_event.set()
            raise RuntimeError(f"Async Run failed: {e}") from e

    def reset(self, a_force: bool = False) -> None:
        """Reset the recorder state.

        Clears accumulated records and optionally resets the IO object.
        If unsaved records are present, raises a warning unless `a_force=True`.

        Args:
            a_force (bool): If True, forcefully clear records even if unsaved.
                If False and unsaved records exist, a warning is raised and
                the reset is aborted.
        """
        if self._records:
            if not a_force:
                warnings.warn(
                    f"{len(self._records)} unsaved records will be lost. " "Use `a_force=True` to force reset.",
                    UserWarning,
                )
                return
            self._records.clear()
