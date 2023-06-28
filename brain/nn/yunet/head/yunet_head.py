""" YuNet Head

"""


# region Imported Dependencies
from typing import List
from torch import nn
# endregion Imported Dependencies


class YuNetHead(nn.Module):
    def __init__(self, a_num_classes: int = 1, a_in_channels: int = 64, a_shared_stacked_convs: int = 1,
                 a_stacked_convs: int = 0, a_feat_channels: int = 64, a_use_kps: bool = True, a_num_kps: int = 5,
                 a_prior_generator_type: str = 'MlvlPointGenerator', a_prior_generator_offset: int = 0,
                 a_prior_generator_strides: List[int] = [8, 16, 32], a_cls_loss_type: str = 'CrossEntropyLoss',
                 a_cls_loss_use_sigmoid: bool = True, a_cls_loss_reduction: str = 'sum',
                 a_cls_loss_loss_weight: float = 1.0, a_bbox_loss_type: str = 'EIoULoss',
                 a_bbox_loss_loss_weight: float = 5.0, a_bbox_loss_reduction: str = 'sum',
                 a_kps_loss_type: str = 'SmoothL1Loss', a_kps_loss_beta: float = 0.1111111111111111,
                 a_kps_loss_loss_weight: float = 0.1, a_obj_loss_type: str = 'CrossEntropyLoss',
                 a_obj_loss_use_sigmoid: bool = True, a_obj_loss_reduction: str = 'sum',
                 a_obj_loss_loss_weight: float = 1.0) -> None:
        super(YuNetHead, self).__init__()
        self.num_classes: int = a_num_classes
        self.in_channels: int = a_in_channels
        self.shared_stacked_convs: int = a_shared_stacked_convs
        self.stacked_convs: int = a_stacked_convs
        self.feat_channels: int = a_feat_channels
        self.use_kps: bool = a_use_kps
        self.num_kps: int = a_num_kps
        self.prior_generator_type: str = a_prior_generator_type
        self.prior_generator_offset: int = a_prior_generator_offset
        self.prior_generator_strides: List[int] = a_prior_generator_strides
        self.cls_loss_type: str = a_cls_loss_type
        self.cls_loss_use_sigmoid: bool = a_cls_loss_use_sigmoid
        self.cls_loss_reduction: str = a_cls_loss_reduction
        self.cls_loss_loss_weight: float = a_cls_loss_loss_weight
        self.bbox_loss_type: str = a_bbox_loss_type
        self.bbox_loss_loss_weight: float = a_bbox_loss_loss_weight
        self.bbox_loss_reduction: str = a_bbox_loss_reduction
        self.kps_loss_type: str = a_kps_loss_type
        self.kps_loss_beta: float = a_kps_loss_beta
        self.kps_loss_loss_weight: float = a_kps_loss_loss_weight
        self.obj_loss_type: str = a_obj_loss_type
        self.obj_loss_use_sigmoid: bool = a_obj_loss_use_sigmoid
        self.obj_loss_reduction: str = a_obj_loss_reduction
        self.obj_loss_loss_weight: float = a_obj_loss_loss_weight

    def forward(self):
        NotImplemented
