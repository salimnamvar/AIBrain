""" YuNet Object Detector Neural Network Model

    This file defines a YuNet object detector neural network model.
"""


# region Imported Dependencies
from brain.cfg.param import Config
from brain.nn.base import BaseNNModel
# endregion Imported Dependencies


class BaseModel(BaseNNModel):
    def __init__(self):
        self.cfg: Config = Config.get_instance()

    def inference(self):
        NotImplemented

    def train(self):
        NotImplemented

    def test(self):
        NotImplemented