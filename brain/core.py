"""AIBrain Core

    This file defines an application core to handle the subsystems and desired pipelines.
"""


# region Imported Dependencies
import importlib
from brain.tsk.base import BaseTask
from brain.cfg.param import Config
# endregion Imported Dependencies


class Sys:
    def __init__(self, a_cfg: str):
        self.cfg: Config = Config.get_instance(a_cfg=a_cfg)
        self.task: BaseTask = None
        self.create_task()

    def create_task(self):
        try:
            module_name = f"brain.tsk.{self.cfg.task.name.lower()}.tsk"
            module = importlib.import_module(module_name)
            task_class = getattr(module, 'Task')
            self.task: BaseTask = task_class()
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `tsk.name` is entered.')

    def run(self):
        try:
            task_method = getattr(self.task, self.cfg.task.method.lower())
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `tsk.method` is entered.')
        task_method()
