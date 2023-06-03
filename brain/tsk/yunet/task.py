""" YuNet Object Detection Research Task

    This task is research in YuNet object detection architecture.
"""


# region Imported Dependencies
from brain.tsk.base import BaseTask
# endregion Imported Dependencies


class Task(BaseTask):
    def __init__(self):
        super().__init__()

    def inference(self, *args, **kwargs):
        NotImplemented

    def train(self, *args, **kwargs):
        NotImplemented

    def test(self, *args, **kwargs):
        NotImplemented
