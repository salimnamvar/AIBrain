"""2D Image

This python file contains the 2D image class object.
"""


# region Imported Dependencies
import numpy as np
# endregion Imported Dependencies


class Image2D(np.ndarray):
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], tuple):
            input_array = np.empty(args[0], **kwargs)
        else:
            input_array = np.array(*args, **kwargs)

        if input_array.ndim in [2, 3]:
            obj = input_array.view(cls)
        else:
            raise ValueError("Input array must be 2D or 3D with at least 1 channel")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def channels(self) -> int:
        if self.ndim == 2:
            return 1
        else:
            return self.shape[2]
