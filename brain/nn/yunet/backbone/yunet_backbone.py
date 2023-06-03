""" YuNet Backbone

"""


# region Imported Dependencies
from typing import List
from torch import nn, Tensor
import torch.nn.functional as F
from brain.nn.yunet.util.yunet_layer import ConvHead, Conv4LayerBlock
# endregion Imported Dependencies


class YuNetBackbone(nn.Module):
    def __init__(self, a_stage_channels: List[List[int]], a_downsample_idx: List[int], a_out_idx: List[int]) -> None:
        super(YuNetBackbone, self).__init__()

        # region Inputs
        self.stage_channels: List[List[int]] = a_stage_channels
        self.downsample_idx: List[int] = a_downsample_idx
        self.out_idx: List[int] = a_out_idx
        # endregion Inputs

        # Initialize Architecture
        self.model0: ConvHead = ConvHead(*self.stage_channels[0])
        for i in range(1, len(self.stage_channels)):
            self.add_module(f'model{i}', Conv4LayerBlock(*self.stage_channels[i]))

        # Initialize
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

    def forward(self, x) -> List[Tensor]:
        out: List[Tensor] = []
        for i in range(len(self.stage_channels)):
            x = self.__getattr__(f'model{i}')(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out


# region Experiment
if __name__ == '__main__':
    if False:
        yunet_backbone = YuNetBackbone(a_stage_channels=[[3, 16, 16], [16, 64], [64, 64], [64, 64], [64, 64], [64, 64]],
                                       a_downsample_idx=[0, 2, 3, 4], a_out_idx=[3, 4, 5])
# endregion Experiment
