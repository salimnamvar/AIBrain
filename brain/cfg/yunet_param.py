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


class PriorGeneratorConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.head.prior_generator.type')
        self.offset: int = int(a_cfg.properties.get('nn.model.head.prior_generator.offset'))
        self.strides: List[int] = a_cfg.properties.get('nn.model.head.prior_generator.strides')


class ClassificationLossConfig:
    def __init__(self, a_cfg: Properties) -> None:
        NotImplemented


class HeadConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.head.type')
        self.num_classes: int = int(a_cfg.properties.get('nn.model.head.num_classes'))
        self.in_channels: int = int(a_cfg.properties.get('nn.model.head.in_channels'))
        self.shared_stacked_convs: int = int(a_cfg.properties.get('nn.model.head.shared_stacked_convs'))
        self.stacked_convs: int = int(a_cfg.properties.get('nn.model.head.stacked_convs'))
        self.feat_channels: int = int(a_cfg.properties.get('nn.model.head.feat_channels'))
        self.prior_generator: PriorGeneratorConfig = PriorGeneratorConfig(a_cfg=a_cfg)


class YuNetConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.backbone: BackboneConfig = BackboneConfig(a_cfg=a_cfg)
        self.neck: NeckConfig = NeckConfig(a_cfg=a_cfg)