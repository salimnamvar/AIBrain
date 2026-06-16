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

    __array_priority__ = 20.0

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

        # Convert Image2D objects to numpy arrays to avoid recursion
        def convert_arg(arg: Any) -> Any:
            if isinstance(arg, Image2D):
                return arg.data
            elif isinstance(arg, (list, tuple)):
                # Handle sequences that might contain Image2D objects
                return type(arg)(convert_arg(item) for item in arg)
            else:
                return arg

        converted_args = tuple(convert_arg(arg) for arg in args)

        result = func(*converted_args, **kwargs)

        if isinstance(result, np.ndarray):
            return Image2D(cast(npt.NDArray[Any], result), filename=self.filename)
        return result

    @property
    def __array_interface__(self):
        return self.data.__array_interface__

    def __getstate__(self) -> Dict[str, Any]:
        return {"data": self.data, "filename": self.filename}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "data", state["data"])
        object.__setattr__(self, "filename", state["filename"])

    def __getitem__(self, key: Any) -> "Image2D":
        sub = self.data[key]
        return Image2D(sub, filename=self.filename)

    # Arithmetic operators
    def __add__(self, other: Any) -> "Image2D":
        """Add operation."""
        if isinstance(other, Image2D):
            result = self.data + other.data
        else:
            result = self.data + other
        return Image2D(result, filename=self.filename)

    def __radd__(self, other: Any) -> "Image2D":
        """Right add operation."""
        result = other + self.data
        return Image2D(result, filename=self.filename)

    def __sub__(self, other: Any) -> "Image2D":
        """Subtract operation."""
        if isinstance(other, Image2D):
            result = self.data - other.data
        else:
            result = self.data - other
        return Image2D(result, filename=self.filename)

    def __rsub__(self, other: Any) -> "Image2D":
        """Right subtract operation."""
        result = other - self.data
        return Image2D(result, filename=self.filename)

    def __mul__(self, other: Any) -> "Image2D":
        """Multiply operation."""
        if isinstance(other, Image2D):
            result = self.data * other.data
        else:
            result = self.data * other
        return Image2D(result, filename=self.filename)

    def __rmul__(self, other: Any) -> "Image2D":
        """Right multiply operation."""
        result = other * self.data
        return Image2D(result, filename=self.filename)

    def __truediv__(self, other: Any) -> "Image2D":
        """True division operation."""
        if isinstance(other, Image2D):
            result = self.data / other.data
        else:
            result = self.data / other
        return Image2D(result, filename=self.filename)

    def __rtruediv__(self, other: Any) -> "Image2D":
        """Right true division operation."""
        result = other / self.data
        return Image2D(result, filename=self.filename)

    # Comparison operators
    def __gt__(self, other: Any) -> "Image2D":
        """Greater than operation."""
        if isinstance(other, Image2D):
            result = self.data > other.data
        else:
            result = self.data > other
        return Image2D(result, filename=self.filename)

    def __ge__(self, other: Any) -> "Image2D":
        """Greater than or equal operation."""
        if isinstance(other, Image2D):
            result = self.data >= other.data
        else:
            result = self.data >= other
        return Image2D(result, filename=self.filename)

    def __lt__(self, other: Any) -> "Image2D":
        """Less than operation."""
        if isinstance(other, Image2D):
            result = self.data < other.data
        else:
            result = self.data < other
        return Image2D(result, filename=self.filename)

    def __le__(self, other: Any) -> "Image2D":
        """Less than or equal operation."""
        if isinstance(other, Image2D):
            result = self.data <= other.data
        else:
            result = self.data <= other
        return Image2D(result, filename=self.filename)

    def __eq__(self, other: Any) -> Any:
        """Equality operation."""
        # For Image2D objects, return element-wise comparison as Image2D
        if isinstance(other, Image2D):
            result = self.data == other.data
            return Image2D(result, filename=self.filename)
        elif isinstance(other, (int, float, np.number)):
            # For scalar comparisons, return element-wise comparison as Image2D
            result = self.data == other
            return Image2D(result, filename=self.filename)
        else:
            # For object identity, use parent behavior
            return super().__eq__(other)

    def __ne__(self, other: Any) -> Any:
        """Not equal operation."""
        # For Image2D objects, return element-wise comparison as Image2D
        if isinstance(other, Image2D):
            result = self.data != other.data
            return Image2D(result, filename=self.filename)
        elif isinstance(other, (int, float, np.number)):
            # For scalar comparisons, return element-wise comparison as Image2D
            result = self.data != other
            return Image2D(result, filename=self.filename)
        else:
            # For object identity, use parent behavior
            return super().__ne__(other)

    # Additional useful methods
    def astype(self, dtype: Any, **kwargs: Any) -> "Image2D":
        """Convert array to a specified type."""
        result = self.data.astype(dtype, **kwargs)
        return Image2D(result, filename=self.filename)

    def flatten(self) -> "Image2D":
        """Return a flattened copy of the array."""
        result = self.data.flatten()
        return Image2D(result, filename=self.filename)

    def reshape(self, *args: Any, **kwargs: Any) -> "Image2D":
        """Return an array with a new shape."""
        result = self.data.reshape(*args, **kwargs)
        return Image2D(result, filename=self.filename)

    def transpose(self, *axes: Any) -> "Image2D":
        """Return an array with axes transposed."""
        result = self.data.transpose(*axes)
        return Image2D(result, filename=self.filename)

    @property
    def T(self) -> "Image2D":
        """Return the transpose of the array."""
        return self.transpose()
