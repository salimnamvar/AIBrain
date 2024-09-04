"""OpenVino Model

This module defines utilities related to the OpenVino-based model.

Classes:
    - OVModel: Represents a base model using the OpenVino library.
    - OVModelList: Represents a container for a collection of OVModel objects.

"""

# region Imported Dependencies
from abc import abstractmethod, ABC
from typing import List, Tuple

from openvino import CompiledModel, Core
from openvino._pyopenvino import Shape

from brain.util.cv.shape import Size
from brain.util.ml.util import BaseModel, BaseModelList
from brain.util.obj import BaseObjectList


# endregion Imported Dependencies


class OVModel(BaseModel, ABC):
    """OpenVino Model

    This class implements a base model using OpenVino library.

    Attributes:
        mdl_path (str): The path to the model.
        mdl_device (str): The device on which the model runs.
        mdl (ExecutableNetwork): The inference executable model.
        mdl_inp_shape (Shape): The input shape of the model.
        mdl_inp_size (Size): The input data size which indicates the width and height.
    """

    def __init__(self, a_name: str, a_mdl_path: str, a_mdl_device: str):
        """Constructor of OpenVino Model

        Initializes a new instance of the OpenVino Model.

        Args:
            a_name (str): The name of the model.
            a_mdl_path (str): The path to the model.
            a_mdl_device (str): The device on which the model runs.
        """
        super().__init__(a_name=a_name)
        self._mdl: CompiledModel = None

        self.mdl_name: str = a_name
        self.mdl_path: str = a_mdl_path
        self.mdl_device: str = a_mdl_device

    # region Attributes
    @property
    def mdl_path(self) -> str:
        """Getter for the model path.

        Returns:
            str: The path of the model.
        """
        return self._mdl_path

    @mdl_path.setter
    def mdl_path(self, a_mdl_path: str) -> None:
        """Setter for the model path.

        Args:
            a_mdl_path (str): The new path for the model.

        Raises:
            TypeError: If `a_mdl_path` is not a string.
        """
        if a_mdl_path is None or not isinstance(a_mdl_path, str):
            raise TypeError("The `a_mdl_path` should be a `str`.")
        self._mdl_path: str = a_mdl_path

    @property
    def mdl_device(self) -> str:
        """Getter for the model device.

        Returns:
            str: The device on which the model is loaded.
        """
        return self._mdl_device

    @mdl_device.setter
    def mdl_device(self, a_mdl_device: str):
        """Setter for the model device.

        Args:
            a_mdl_device (str): The new device for the model.

        Raises:
            TypeError: If `a_mdl_device` is not a string.
        """
        if a_mdl_device is None or not isinstance(a_mdl_device, str):
            raise TypeError("The `a_mdl_device` must be a `str`.")
        self._mdl_device: str = a_mdl_device

    @property
    def mdl(self) -> CompiledModel:
        """Getter for the model.

        Returns:
            CompiledModel: The model.
        """
        return self._mdl

    @mdl.setter
    def mdl(self, a_mdl: CompiledModel) -> None:
        """Setter for the model.

        Args:
            a_mdl (CompiledModel): The new model.

        Raises:
            TypeError: If `a_mdl` is not an instance of `CompiledModel`.
        """
        if a_mdl is None or not isinstance(a_mdl, CompiledModel):
            raise TypeError("The `a_mdl` should be a `CompiledModel`.")
        self._mdl: CompiledModel = a_mdl

    @property
    def mdl_inp_shape(self) -> Tuple[int, ...]:
        """Getter for the input shape of the model.

        Returns:
            Tuple[int, ...]: The input shape of the model.

        Raises:
            TypeError: If the model is not loaded.
        """
        self.validate_mdl()
        try:
            # Attempt to retrieve the partial shape of the first input
            partial_shape = self.mdl.inputs[0].get_partial_shape()

            if partial_shape.is_static:
                # If the shape is static, return as Shape
                return tuple(partial_shape.to_shape())
            else:
                # If the shape is dynamic, return as PartialShape
                return tuple(d.get_length() if d.is_static else None for d in partial_shape)
        except RuntimeError:
            raise TypeError("Unable to retrieve input shape due to a dynamic shape error.")
        except Exception as e:
            raise TypeError(f"Unexpected error when retrieving input shape: {e}")

    @property
    def mdl_inp_size(self) -> Size:
        """Getter for the input size of the model.

        Returns:
            Size: The input size of the model.

        Raises:
            TypeError: If the model is not loaded.
        """
        self.validate_mdl()
        return Size(a_height=self.mdl_inp_shape[2], a_width=self.mdl_inp_shape[3])

    # endregion Attributes

    def to_dict(self) -> dict:
        """Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the `OVModel`.
        """
        dic = {"name": self.name, "mdl_path": self.mdl_path, "mdl_device": self.mdl_device}
        return dic

    def validate_mdl(self) -> None:
        """Validate the model.

        Raises:
            TypeError: If the model is not loaded.
        """
        if not (hasattr(self, "_mdl") and isinstance(self._mdl, CompiledModel)):
            raise TypeError("The mdl must be loaded by `load_mdl()` method.")

    def load_mdl(self) -> None:
        """Load Model

        It loads and initializes the model.

        Raises:
            TypeError: If the model is not loaded successfully.
        """
        core = Core()
        net = core.read_model(self.mdl_path)
        self._mdl = core.compile_model(net, self._mdl_device)

    @abstractmethod
    def _preproc(self, *args, **kwargs):
        """Pre-process input data.

        This method should be implemented by subclasses to define the pre-processing
        steps for the input data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `_preproc`")

    @abstractmethod
    def _postproc(self, *args, **kwargs):
        """Post-process predictions.

        This method should be implemented by subclasses to define the post-processing
        steps for the model predictions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `_postproc`")

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Run inference on the model.

        This method must be implemented by subclasses.

        Args:
            *args: Variable-length arguments.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `_postproc`")


class OVModelList(BaseModelList, BaseObjectList[OVModel], ABC):
    """OpenVino Model List

    The OVModelList class is based on the :class:`BaseObjectList` class and serves as a container for a collection of
    :class:`OVModel` objects.

    Attributes:
        For inherited attribute details, see `BaseModelList` class documentation.
    """

    def __init__(
        self,
        a_name: str = "OVModelList",
        a_max_size: int = -1,
        a_items: List[OVModel] = None,
    ):
        """
        Constructor for the `OVModelList` class.

        Args:
            a_name (str, optional):
                A :type:`string` that specifies the name of the `OVModelList` instance (default is 'OVModelList').
            a_max_size (int, optional):
                An :type:`int` representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[OVModel], optional):
                A list of :class:`OVModel` objects to initialize the `OVModelList` (default is None).

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    def load_mdl(self) -> None:
        """Load Model

        This method iterates through the list and loads and initializes the models.

        Returns:
            None
        """
        for mdl in self.items:
            mdl.load_mdl()
