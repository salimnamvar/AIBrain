""" YuNet Detector Architecture

    This file defines the YuNet Detector Architecture.
"""

# region Imported Dependencies
import torch.nn as nn
from brain.cfg.param import Config
from brain.nn.yunet.backbone.yunet_backbone import YuNetBackbone
from brain.nn.yunet.neck.tfpn import TFPN
# endregion Imported Dependencies


class YuNetDetector(nn.Module):
    def __init__(self):
        super(YuNetDetector, self).__init__()
        self.cfg: Config = Config.get_instance()

        if self.cfg.nn.model.backbone.type == 'YuNetBackbone':
            self.backbone: YuNetBackbone = YuNetBackbone(a_stage_channels=self.cfg.nn.model.backbone.stage_channels,
                                                         a_downsample_idx=self.cfg.nn.model.backbone.downsample_idx,
                                                         a_out_idx=self.cfg.nn.model.backbone.out_idx)
        else:
            self.backbone: YuNetBackbone = YuNetBackbone(a_stage_channels=self.cfg.nn.model.backbone.stage_channels,
                                                         a_downsample_idx=self.cfg.nn.model.backbone.downsample_idx,
                                                         a_out_idx=self.cfg.nn.model.backbone.out_idx)
        if self.cfg.nn.model.neck.type == 'TFPN':
            self.neck: TFPN = TFPN(a_in_channels=self.cfg.nn.model.neck.in_channels,
                                   a_out_idx=self.cfg.nn.model.neck.out_idx)
        else:
            self.neck: TFPN = TFPN(a_in_channels=self.cfg.nn.model.neck.in_channels,
                                   a_out_idx=self.cfg.nn.model.neck.out_idx)

        # TODO: Add the Bounding Box Head

    def forward(self, x):
        NotImplemented
