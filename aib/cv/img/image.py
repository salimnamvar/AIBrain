"""Computer Vision - Image - Image Data Class Utilities

This module provides the Image2D data class, which represents a 2D image with associated metadata.

Classes:
    - Image2D: Represents a 2D image with metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

from aib.cnt.b_data import BaseData
from aib.cv.geom.size import Size


@dataclass(frozen=True)
class Image2D(BaseData):
    """Image2D Data Class

    Represents a 2D image with associated metadata. It includes properties for width, height, size,
    aspect ratio, and number of channels.

    Attributes:
        data (npt.NDArray[Any]): The image data as a NumPy array.
        filename (Optional[str]): Optional filename associated with the image.
        width (int): Width of the image.
        height (int): Height of the image.
        size (Size): Size of the image.
        aspect_ratio (float): Aspect ratio of the image.
        channels (int): Number of channels in the image.
    """

    data: npt.NDArray[Any] = field(compare=False)
    filename: Optional[str] = field(default=None, compare=False)

    @property
    def width(self) -> int:
        """Width of the image.

        Returns:
            int: Width of the image.
        """
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Height of the image.

        Returns:
            int: Height of the image.
        """
        return self.data.shape[0]

    @property
    def size(self) -> Size[int]:
        """Size of the image.

        Returns:
            Size[int]: A Size object containing the width and height of the image.
        """
        return Size[int](width=int(self.width), height=int(self.height))

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the image.

        Returns:
            float: Aspect ratio of the image (width / height).
        """
        return self.size.aspect_ratio

    @property
    def channels(self) -> int:
        """Number of channels in the image.

        Returns:
            int: Number of channels in the image. Returns 1 for grayscale images.
        """
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert the image data to a NumPy array.

        Returns:
            npt.NDArray[Any]: The image data as a NumPy array.
        """
        return self.data

    def __array__(self, dtype: Optional[np.dtype[Any]] = None, copy: Optional[bool] = None):
        if dtype is not None:
            arr = self.to_numpy().astype(dtype, copy=copy if copy is not None else False)
        else:
            arr = self.to_numpy().copy() if copy else self.to_numpy()
        return arr

    def __array_wrap__(
        self,
        array: npt.NDArray[Any],
        _context: Optional[Tuple[Any, tuple[Any, ...], int]] = None,
        return_scalar: bool = False,
    ) -> Any:
        if return_scalar:
            return array.item()

        if array.ndim == 0:
            return array.item()

        return Image2D(data=array, filename=self.filename)

    def __array_finalize__(self, obj: Optional[object]) -> None:
        if obj is None:
            return
        object.__setattr__(self, "filename", getattr(obj, "filename", None))

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        arrays = [np.asarray(x) if isinstance(x, Image2D) else x for x in inputs]

        if "out" in kwargs:
            out = kwargs["out"]
            kwargs["out"] = tuple(np.asarray(x) if isinstance(x, Image2D) else x for x in out)

        result: Any = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, np.ndarray):
            return Image2D(cast(npt.NDArray[Any], result), filename=self.filename)
        elif isinstance(result, tuple):
            result_tuple: Tuple[Any, ...] = cast(Tuple[Any, ...], result)
            wrapped: Tuple[Any, ...] = tuple(
                Image2D(cast(npt.NDArray[Any], x), filename=self.filename) if isinstance(x, np.ndarray) else x
                for x in result_tuple
            )
            return wrapped
        else:
            return result

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Sequence[type],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if not all(issubclass(t, (np.ndarray, Image2D)) for t in types):
            return NotImplemented

        arrays = [np.asarray(a) if isinstance(a, Image2D) else a for a in args]

        result = func(*arrays, **kwargs)

        if isinstance(result, np.ndarray):
            return Image2D(cast(npt.NDArray[Any], result), filename=self.filename)
        return result
