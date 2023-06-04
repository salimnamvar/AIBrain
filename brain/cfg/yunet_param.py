"""Configuration Parameters

    This file contains the struct classes of defining the configuration parameters related to the YuNet Object Detector.
"""


# region Imported Dependencies
from typing import List
from jproperties import Properties
# endregion Imported Dependencies


class BackboneConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.backbone.type')
        self.stage_channels: List[List[int]] = eval(a_cfg.properties.get('nn.model.backbone.stage_channels'))
        self.downsample_idx: List[int] = eval(a_cfg.properties.get('nn.model.backbone.downsample_idx'))
        self.out_idx: List[int] = eval(a_cfg.properties.get('nn.model.backbone.out_idx'))


class NeckConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.neck.type')
        self.in_channels: List[int] = eval(a_cfg.properties.get('nn.model.neck.in_channels'))
        self.out_idx: List[int] = eval(a_cfg.properties.get('nn.model.neck.out_idx'))


class YuNetConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.backbone: BackboneConfig = BackboneConfig(a_cfg=a_cfg)
        self.neck: NeckConfig = NeckConfig(a_cfg=a_cfg)