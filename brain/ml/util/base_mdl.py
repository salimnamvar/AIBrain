"""Base Model

This module defines utilities pertaining to a foundational model capable of seamlessly operating in inference, training,
and testing modes.

Classes:
    - BaseModel: Base class for models capable of performing inference, training, and testing.
    - BaseModelList: Container class for a collection of BaseModel objects.
"""

# region Imported Dependencies
import logging
from abc import abstractmethod, ABC
from typing import List

from brain.cfg import BrainConfig
from brain.obj import ExtBaseObject, BaseObjectList


# endregion Imported Dependencies


class BaseModel(ExtBaseObject, ABC):
    """Base Model

    The base class for models capable of performing inference, training, and testing.

    Attributes:
        name (str): The name of the model.
        cfg (BrainConfig, optional): The configuration instance.
        logger (logging.Logger, optional): The logger instance for the model.
    """

    def __init__(self, a_name: str = "BaseModel", a_use_cfg: bool = True):
        """BaseModel Constructor

        Args:
            a_name (str): The name of the model.
            a_use_cfg (bool): Flag to determine if the model should have a cfg attribute.
        """
        super().__init__(a_name)
        self._cfg = None
        self._logger = None

        if a_use_cfg:
            self.cfg: BrainConfig = BrainConfig.get_instance()
            self.logger = logging.getLogger(self.cfg.log.name + "." + self.name)
        else:
            self.logger = logging.getLogger(self.name)

    # region Attributes
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self._logger

    @logger.setter
    def logger(self, a_logger: logging.Logger) -> None:
        """Set the logger instance.

        Args:
            a_logger (logging.Logger): The new logger instance.

        Raises:
            TypeError: If `a_logger` is not an instance of logging.Logger.
        """
        if a_logger is None or not isinstance(a_logger, logging.Logger):
            raise TypeError("The `a_logger` must be a `logging.Logger`.")
        self._logger: logging.Logger = a_logger

    @property
    def cfg(self) -> BrainConfig:
        """Get the BrainConfig instance.

        Returns:
            BrainConfig: The BrainConfig instance.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, a_cfg: BrainConfig) -> None:
        """Set the BrainConfig instance.

        Args:
            a_cfg (BrainConfig): The new BrainConfig instance.

        Raises:
            TypeError: If `a_cfg` is not an instance of BrainConfig.
        """
        if a_cfg is None or not isinstance(a_cfg, BrainConfig):
            raise TypeError("The `a_cfg` must be a `BrainConfig`.")
        self._cfg: BrainConfig = a_cfg

    # endregion Attributes

    def to_dict(self) -> dict:
        """Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the `BaseModel`.
        """
        dic = {"name": self.name}
        return dic

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Inference Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of making inferences
        based n the functionality.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `infer`")

    def train(self, *args, **kwargs):
        """Training Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of training.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `train`")

    def test(self, *args, **kwargs):
        """Testing Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of testing.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.`
        """
        NotImplementedError("Subclasses must implement `test`")


class BaseModelList(BaseObjectList[BaseModel], ABC):
    """BaseModel List

    The BaseModelList class is based on the :class:`BaseObjectList` class and serves as a container for a collection of
    :class:`BaseModel` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the BaseModelList (default is 'BaseModelList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[BaseModel], optional):
            A list of BaseModel objects to initialize the BaseModelList (default is None).
        cfg: BrainConfig
            An instance of the BrainConfig class for configuration settings.
        logger: logging.Logger
            A :class:`logging.Logger` instance for logging.
    """

    def __init__(
        self,
        a_name: str = "BaseModelList",
        a_max_size: int = -1,
        a_items: List[BaseModel] = None,
        a_use_cfg: bool = False,
    ):
        """
        Constructor for the `BaseModelList` class.

        Args:
            a_name (str, optional):
                A :type:`string` that specifies the name of the `BaseModelList` instance (default is 'BaseModelList').
            a_max_size (int, optional):
                An :type:`int` representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[BaseModel], optional):
                A list of :class:`BaseModel` objects to initialize the `BaseModelList` (default is None).
            a_use_cfg (bool): Flag to determine if the model should have a cfg attribute.

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

        self._cfg = None
        self._logger = None

        if a_use_cfg:
            self.cfg: BrainConfig = BrainConfig.get_instance()
            self.logger = logging.getLogger(self.cfg.log.name + "." + self.name)
        else:
            self.logger = logging.getLogger(self.name)

    # region Attributes
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self._logger

    @logger.setter
    def logger(self, a_logger: logging.Logger) -> None:
        """Set the logger instance.

        Args:
            a_logger (logging.Logger): The new logger instance.

        Raises:
            TypeError: If `a_logger` is not an instance of logging.Logger.
        """
        if a_logger is None or not isinstance(a_logger, logging.Logger):
            raise TypeError("The `a_logger` must be a `logging.Logger`.")
        self._logger: logging.Logger = a_logger

    @property
    def cfg(self) -> BrainConfig:
        """Get the BrainConfig instance.

        Returns:
            BrainConfig: The BrainConfig instance.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, a_cfg: BrainConfig) -> None:
        """Set the BrainConfig instance.

        Args:
            a_cfg (BrainConfig): The new BrainConfig instance.

        Raises:
            TypeError: If `a_cfg` is not an instance of BrainConfig.
        """
        if a_cfg is None or not isinstance(a_cfg, BrainConfig):
            raise TypeError("The `a_cfg` must be a `BrainConfig`.")
        self._cfg: BrainConfig = a_cfg

    # endregion Attributes

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Inference Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of making inferences
        based n the functionality.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `infer`")

    def train(self, *args, **kwargs):
        """Training Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of training.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `train`")

    def test(self, *args, **kwargs):
        """Testing Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of testing.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.`
        """
        NotImplementedError("Subclasses must implement `test`")
