""" Base Task

    This file defines an abstract base brain task.
"""


# region Imported Dependencies
from abc import ABC, abstractmethod
from brain.cfg.param import Config
# endregion Imported Dependencies


class BaseTask(ABC):
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
