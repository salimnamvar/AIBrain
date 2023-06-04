""" YuNet TFPN Neck

    Tiny Feature Pyramid Network (TFPN) neck
"""


# region Imported Dependencies
from typing import List
from torch import nn, Tensor
import torch.nn.functional as F
from brain.nn.yunet.util.yunet_layer import ConvDPUnit
# endregion Imported Dependencies


class TFPN(nn.Module):
    def __init__(self, a_in_channels: List[int], a_out_idx: List[int]) -> None:
        super(TFPN, self).__init__()

        # region Inputs
        self.in_channels: List[int] = a_in_channels
        self.out_idx: List[int] = a_out_idx
        self.num_layers: int = len(self.in_channels)
        # endregion Inputs

        # Initialize Architecture
        self.lateral_convs: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(module=ConvDPUnit(a_in_channels=self.in_channels[i],
                                                        a_out_channels=self.in_channels[i],
                                                        a_with_bn_relu=True))
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> List[Tensor]:
        num_features: int = len(x)

        # Top-down flow
        for i in range(num_features - 1, 0, -1):
            x[i] = self.lateral_convs[i](x[i])
            x[i - 1] = x[i - 1] + F.interpolate(input=x[i], scale_factor=2., mode='nearest')

        x[0] = self.lateral_convs[0](x[0])

        outs = [x[i] for i in self.out_idx]
        return outs


# region Experiment
if __name__ == '__main__':
    if True:
        yunet_tfpn = TFPN(a_in_channels=[64, 64, 64], a_out_idx=[0, 1, 2])
# endregion Experiment
