""" Base Machine Learning Model

    This file defines a base machine learning model.
"""


# region Imported Dependencies
from abc import ABC, abstractmethod
from brain.cfg.param import Config
# endregion Imported Dependencies


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        self.cfg: Config = Config.get_instance()

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass
