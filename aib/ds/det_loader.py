"""Dataset - Detection Dataset Loader Utilities

This module provides utilities for loading and managing person-detection datasets stored in Parquet format.
It defines data structures and loader classes for efficient access to frame-level detection results,
including bounding box information.

Classes:
    FrameDetections:
        Data class representing detection results for a single frame, including the frame and its bounding boxes.
    DetectionFileLoader:
        Loader for a single Parquet file containing detection results.
    DetectionDatasetLoader:
        Loader for a argsory of Parquet detection files, managing multiple DetectionFileLoader instances.
"""

import time
from dataclasses import dataclass, field
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from aib.cnt.b_data import BaseData
from aib.cnt.b_dict import BaseDict
from aib.cv.geom.box.bbox2d import FloatBBox2DList
from aib.cv.img.frame import Frame2D
from aib.sys.b_obj import BaseObject


@dataclass(frozen=True)
class FrameDetections(BaseData):
    """Frame Detections Data Class

    Attributes:
        frame (Frame2D): The frame
        boxes (FloatBBox2DList): The 2D bounding box list detection results
    """

    frame: Frame2D = field(compare=False, metadata={"description": "The frame"})
    boxes: Optional[FloatBBox2DList] = field(
        compare=False, metadata={"description": "The 2D bounding box list detection results"}
    )


class DetectionFileLoader(BaseObject):
    """Detection File Loader

    This class is responsible for loading and processing a single Parquet file containing
    person detection information.

    Attributes:
        _file_path (Path): The file path to the dataset.
        _in_memory (bool): Whether to load the dataset into memory.
        _frame_cols (List[str]): The columns to use for frame information.
        _box_cols (List[str]): The columns to use for bounding box information.
        _dataset (Optional[ds.Dataset | pd.DataFrame]): The loaded dataset.
        _frame_ids (Optional[List[int]]): The list of frame IDs in the dataset.
    """

    def __init__(
        self,
        a_id: int,
        a_file_path: str | PathLike[str],
        a_in_memory: bool = False,
        a_frame_cols: Tuple[str, ...] = ("frame_id", "frame_timestamp", "frame_src_name"),
        a_box_cols: Tuple[str, ...] = ("box_x1", "box_y1", "box_x2", "box_y2", "box_score", "box_label"),
        a_name: str = 'ParquetDetDataset',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Parquet Dataset Loader Initialization

        Args:
            a_id (int): The ID of the dataset.
            a_file_path (str | PathLike[str]): The file path to the dataset.
            a_in_memory (bool, optional): Whether to load the dataset into memory. Defaults to False.
            a_frame_cols (Tuple[str], optional):
                The columns to use for frame information. Defaults to
                ["frame_id", "frame_timestamp", "frame_src_name"].
            a_box_cols (Tuple[str], optional):
                The columns to use for bounding box information. Defaults to
                ["box_x1", "box_y1", "box_x2", "box_y2", "box_score", "box_label"].
            a_name (str, optional): The name of the dataset loader. Defaults to 'ParquetDetDataset'.
            a_use_prof (bool, optional): Whether to enable profiling. Defaults to False.
            a_use_cfg (bool, optional): Whether to enable configuration. Defaults to True.
            a_use_log (bool, optional): Whether to enable logging. Defaults to True.
            **kwargs (Dict[str, Any], optional): Additional keyword arguments.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._file_path: Path = Path(a_file_path)
        self._in_memory: bool = a_in_memory
        self._frame_cols: List[str] = list(a_frame_cols)
        self._box_cols: List[str] = list(a_box_cols)
        self._dataset: Optional[ds.Dataset | pd.DataFrame] = None
        self._frame_ids: Optional[List[int]] = None

    def load(self) -> None:
        """Load the dataset from the specified file path.

        Raises:
            ValueError: If the dataset cannot be loaded.
        """
        try:
            if self._in_memory:
                self._dataset: pd.DataFrame = pd.read_parquet(self._file_path)
                self._frame_ids = sorted(self._dataset["frame_id"].unique())
                self._dataset.set_index("frame_id", inplace=True, drop=False)
            else:
                self._dataset: ds.Dataset = ds.dataset(str(self._file_path), format='parquet')
                table: pa.Table = self._dataset.to_table(columns=["frame_id"])
                self._frame_ids = sorted(set(table["frame_id"].to_pylist()))
        except Exception as e:
            self._frame_ids = None
            raise ValueError(f"Failed to load dataset, file: {self._file_path}, error: {e}") from e

    def _df_to_det(self, a_id: int, a_df: pd.DataFrame) -> FrameDetections:
        """Convert a DataFrame to a Detections object.

        Args:
            a_id (int): The frame ID.
            a_df (pd.DataFrame): The DataFrame containing detection information.

        Returns:
            Detections: The Detections object.
        """
        if a_df.empty:
            return FrameDetections(
                frame=Frame2D(
                    data=np.empty((768, 1024, 3), dtype=np.uint8),
                    filename='',
                    id=a_id,
                    timestamp=time.time(),
                    src_id=self.id,
                    src_uri=None,
                    src_name=None,
                ),
                boxes=None,
            )

        src_name = a_df['frame_src_name'].iloc[0]
        timestamp = a_df['frame_timestamp'].iloc[0]
        width = a_df['frame_width'].iloc[0]
        height = a_df['frame_height'].iloc[0]
        frame = Frame2D(
            data=np.empty((height, width, 3), dtype=np.uint8),
            filename='',
            id=a_id,
            src_id=self.id,
            timestamp=timestamp,
            src_name=src_name,
        )
        if a_df[self._box_cols].isna().all(axis=1).all():
            return FrameDetections(frame=frame, boxes=FloatBBox2DList())

        return FrameDetections(
            frame=frame,
            boxes=FloatBBox2DList.from_xyxy(a_df[self._box_cols].to_numpy(dtype=np.float32), a_use_float=True),
        )

    def __getitem__(self, a_id: int) -> FrameDetections:
        """Get detection information for a specific frame ID.

        Args:
            a_id (int): The frame ID to retrieve detection information for.

        Returns:
            Detections: The detection information for the specified frame ID.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        if self._in_memory:
            df: pd.DataFrame = self._dataset.loc[a_id]
            if isinstance(df, pd.Series):
                df = df.to_frame().T
        else:
            table = self._dataset.to_table(filter=pc.field("frame_id") == a_id)
            df: pd.DataFrame = table.to_pandas()
        return self._df_to_det(a_id=a_id, a_df=df)

    def __len__(self) -> int:
        """Get the number of frames in the dataset.

        Returns:
            int: The number of frames in the dataset.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._frame_ids is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        return len(self._frame_ids)

    def __iter__(self) -> Iterator[Tuple[int, FrameDetections]]:
        """Iterate over the dataset and yield frame IDs and their corresponding detection information.

        Yields:
            Tuple[int, Detections]: A tuple containing the frame ID and its detection information.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._dataset is None or self._frame_ids is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        for fid in self._frame_ids:
            yield fid, self[fid]

    def release(self) -> None:
        """Release resources held by the dataset."""
        if self._dataset is None or self._frame_ids is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")

        self._dataset = None
        self._frame_ids = None


class DetectionDatasetLoader(BaseObject):
    """Detection Dataset Loader

    This class is responsible for loading and managing the detection dataset.

    Attributes:
        _dataset_dir (Path): The argsory containing the dataset files.
        _file_ext (Sequence[str]): The file extension for the dataset files.
        _in_memory (bool): Whether to load the dataset into memory.
        _frame_cols (Tuple[str, ...]): The columns to use for frame metadata.
        _box_cols (Tuple[str, ...]): The columns to use for bounding box metadata.
        loaders (Optional[BaseDict[str, ParquetFileLoader]]): The loaders for the dataset files.
    """

    def __init__(
        self,
        a_dataset_dir: str | PathLike[str],
        a_file_ext: Sequence[Literal["parquet", "csv"]] = ["parquet"],
        a_in_memory: bool = False,
        a_frame_cols: Tuple[str, ...] = (
            "frame_id",
            "frame_timestamp",
            "frame_src_name",
            "frame_width",
            "frame_height",
        ),
        a_box_cols: Tuple[str, ...] = ("box_x1", "box_y1", "box_x2", "box_y2", "box_score", "box_label"),
        a_id: Optional[int] = None,
        a_name: str = 'DetectionDatasetLoader',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset loader.

        Args:
            a_dataset_dir (str | PathLike[str]): The argsory containing the dataset files.
            a_file_ext (Sequence[Literal["parquet", "csv"]]): The file extension for the dataset files. Defaults to ["parquet"].
            a_in_memory (bool, optional): Whether to load the dataset into memory. Defaults to False.
            a_frame_cols (Tuple[str, ...], optional):
                The columns to use for frame metadata. Defaults to ("frame_id", "frame_timestamp", "frame_src_name").
            a_box_cols (Tuple[str, ...], optional):
                The columns to use for bounding box metadata. Defaults to
                ("box_x1", "box_y1", "box_x2", "box_y2", "box_score", "box_label").
            a_id (Optional[int], optional): An optional ID for the dataset loader. Defaults to None.
            a_name (str, optional): The name of the dataset loader. Defaults to 'DetectionDatasetLoader'.
            a_use_prof (bool, optional): Whether to enable profiling. Defaults to False.
            a_use_cfg (bool, optional): Whether to enable configuration. Defaults to True.
            a_use_log (bool, optional): Whether to enable logging. Defaults to True.
            **kwargs (Dict[str, Any], optional): Additional keyword arguments.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._in_memory: bool = a_in_memory
        self._frame_cols: Tuple[str, ...] = a_frame_cols
        self._box_cols: Tuple[str, ...] = a_box_cols
        self._file_ext: Sequence[str] = a_file_ext
        self._dataset_dir: Path = Path(a_dataset_dir).resolve()
        self._loaders: Optional[BaseDict[str, DetectionFileLoader]] = None

    def load(self) -> None:
        """Load the dataset.

        Loads the parquet files from the dataset argsory and creates loaders for each file.
        """
        self._loaders = BaseDict[str, DetectionFileLoader]()
        for i, path in enumerate(
            sorted(
                chain.from_iterable(
                    self._dataset_dir.rglob(f"*.{ext}")
                    for ext in {ext.lower() for ext in self._file_ext} | {ext.upper() for ext in self._file_ext}
                )
            )
        ):
            rel_path = str(path.relative_to(self._dataset_dir))
            loader = DetectionFileLoader(
                a_id=i,
                a_file_path=str(path),
                a_in_memory=self._in_memory,
                a_box_cols=self._box_cols,
                a_frame_cols=self._frame_cols,
                a_name=rel_path,
            )
            self._loaders[rel_path] = loader

    @property
    def loaders(self) -> Optional[BaseDict[str, DetectionFileLoader]]:
        """Get the parquet file loaders.

        Returns:
            Optional[BaseDict[str, ParquetFileLoader]]: The parquet file loaders.
            Optional[BaseDict[str, ParquetFileLoader]]: The parquet file loaders.
        """
        return self._loaders

    def release(self) -> None:
        """Release resources held by the dataset.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        for video_reader in self._loaders.values():
            video_reader.release()
        self._loaders.clear()

    def __len__(self) -> int:
        """Get the number of parquet file loaders.

        Returns:
            int: The number of parquet file loaders.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        return len(self._loaders)

    def __iter__(self) -> Iterator[Tuple[str, DetectionFileLoader]]:
        """Get an iterator over the parquet file loaders.

        Returns:
            Iterator[Tuple[str, ParquetFileLoader]]: An iterator over the parquet file loaders.

        Raises:
            ValueError: If the dataset is not loaded.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        return iter(self._loaders.items())

    def __getitem__(self, a_key: str) -> DetectionFileLoader:
        """Get a parquet file loader by its key.

        Args:
            a_key (str): The key of the parquet file loader to retrieve.

        Returns:
            ParquetFileLoader: The parquet file loader instance associated with the key.

        Raises:
            ValueError: If the dataset is not loaded or the key is not found.
        """
        if self._loaders is None:
            raise ValueError("Dataset not loaded. Call load() before accessing the dataset.")
        return self._loaders[a_key]
