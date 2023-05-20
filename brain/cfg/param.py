"""Configuration Parameters

    This file contains the struct classes of defining the configuration parameters.
"""


# region Imported Dependencies
from jproperties import Properties
# endregion Imported Dependencies


# region Sub Configuration Data Classes
class ExperimentConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.name: str = a_cfg.properties.get('cfg.name')
        self.desc: str = a_cfg.properties.get('cfg.desc')


class TaskConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.name: str = a_cfg.properties.get('tsk.name')
        self.method: str = a_cfg.properties.get('tsk.method')
        if not self.method.lower() in ['train', 'test', 'inference']:
            raise ValueError('Invalid `tsk.method` is entered.')
# endregion Sub Configuration Data Classes


class Config:
    __instance = None

    def __init__(self, a_cfg: str) -> None:
        if Config.__instance is not None:
            raise Exception('The `Config` class is allowed to have one instance.')
        else:
            self.__cfg_path: str = a_cfg

            # Read Configuration File
            cfg: Properties = Properties()
            with open(self.__cfg_path, 'rb') as f:
                cfg.load(f)

            # Sub Configurations
            self.common: ExperimentConfig = ExperimentConfig(a_cfg=cfg)
            self.task: TaskConfig = TaskConfig(a_cfg=cfg)
            Config.__instance = self

    @staticmethod
    def get_instance(a_cfg: str = None):
        if Config.__instance is None:
            Config(a_cfg=a_cfg)
        return Config.__instance
