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
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.prior_generator.type')
        self.offset: int = int(a_cfg.properties.get('nn.model.bbox_head.prior_generator.offset'))
        self.strides: List[int] = eval(a_cfg.properties.get('nn.model.bbox_head.prior_generator.strides'))


class ClsLossConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.cls_loss.type')
        self.use_sigmoid: bool = eval(a_cfg.properties.get('nn.model.bbox_head.cls_loss.use_sigmoid'))
        self.reduction: str = a_cfg.properties.get('nn.model.bbox_head.cls_loss.reduction')
        self.loss_weight: float = eval(a_cfg.properties.get('nn.model.bbox_head.cls_loss.loss_weight'))


class BboxLossConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.bbox_loss.type')
        self.loss_weight: float = float(a_cfg.properties.get('nn.model.bbox_head.bbox_loss.loss_weight'))
        self.reduction: str = a_cfg.properties.get('nn.model.bbox_head.bbox_loss.reduction')


class KpsLossConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.kps_loss.type')
        self.beta: float = float(a_cfg.properties.get('nn.model.bbox_head.kps_loss.beta'))
        self.loss_weight: float = float(a_cfg.properties.get('nn.model.bbox_head.kps_loss.loss_weight'))


class ObjLossConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.obj_loss.type')
        self.use_sigmoid: bool = eval(a_cfg.properties.get('nn.model.bbox_head.obj_loss.use_sigmoid'))
        self.reduction: str = a_cfg.properties.get('nn.model.bbox_head.obj_loss.reduction')
        self.loss_weight: float = float(a_cfg.properties.get('nn.model.bbox_head.obj_loss.loss_weight'))


class HeadConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.type: str = a_cfg.properties.get('nn.model.bbox_head.type')
        self.num_classes: int = int(a_cfg.properties.get('nn.model.bbox_head.num_classes'))
        self.in_channels: int = int(a_cfg.properties.get('nn.model.bbox_head.in_channels'))
        self.shared_stacked_convs: int = int(a_cfg.properties.get('nn.model.bbox_head.shared_stacked_convs'))
        self.stacked_convs: int = int(a_cfg.properties.get('nn.model.bbox_head.stacked_convs'))
        self.feat_channels: int = int(a_cfg.properties.get('nn.model.bbox_head.feat_channels'))
        self.use_kps: bool = eval(a_cfg.properties.get('nn.model.bbox_head.use_kps'))
        self.num_kps: int = int(a_cfg.properties.get('nn.model.bbox_head.num_kps'))
        self.prior_generator: PriorGeneratorConfig = PriorGeneratorConfig(a_cfg=a_cfg)
        self.cls_loss: ClsLossConfig = ClsLossConfig(a_cfg=a_cfg)
        self.bbox_loss: BboxLossConfig = BboxLossConfig(a_cfg=a_cfg)
        self.kps_loss: KpsLossConfig = KpsLossConfig(a_cfg=a_cfg)
        self.obj_loss: ObjLossConfig = ObjLossConfig(a_cfg=a_cfg)


class YuNetConfig:
    def __init__(self, a_cfg: Properties) -> None:
        self.backbone: BackboneConfig = BackboneConfig(a_cfg=a_cfg)
        self.neck: NeckConfig = NeckConfig(a_cfg=a_cfg)
        self.bbox_head: HeadConfig = HeadConfig(a_cfg=a_cfg)