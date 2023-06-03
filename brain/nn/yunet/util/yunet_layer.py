""" YuNet Layer Utility

"""


# region Imported Dependencies
from torch import nn, Tensor
# endregion Imported Dependencies


class ConvDPUnit(nn.Module):
    def __init__(self, a_in_channels: int, a_out_channels: int, a_with_bn_relu: bool) -> None:
        super(ConvDPUnit, self).__init__()

        # region Inputs
        self.in_channels: int = a_in_channels
        self.out_channels: int = a_out_channels
        self.with_bn_relu: bool = a_with_bn_relu
        # endregion Inputs

        # region Layers
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                          stride=1, padding=0, bias=True, groups=1)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3,
                                          stride=1, padding=1, bias=True, groups=self.out_channels)
        if self.with_bn_relu:
            self.bn: nn.BatchNorm2d = nn.BatchNorm2d(self.out_channels)
            self.relu: nn.ReLU = nn.ReLU(inplace=True)
        # endregion Layers

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.conv1(x)
        x: Tensor = self.conv2(x)
        if self.with_bn_relu:
            x: Tensor = self.bn(x)
            x: Tensor = self.relu(x)
        return x


class Conv4LayerBlock(nn.Module):
    def __init__(self, a_in_channels: int, a_out_channels: int, a_with_bn_relu: bool = True) -> None:
        super(Conv4LayerBlock, self).__init__()

        # region Inputs
        self.in_channels: int = a_in_channels
        self.out_channels: int = a_out_channels
        self.with_bn_relu: bool = a_with_bn_relu
        # endregion Inputs

        # region Layers
        self.conv1 = ConvDPUnit(a_in_channels=self.in_channels, a_out_channels=self.in_channels,
                                a_with_bn_relu=self.with_bn_relu)
        self.conv2 = ConvDPUnit(a_in_channels=self.in_channels, a_out_channels=self.out_channels,
                                a_with_bn_relu=self.with_bn_relu)
        # endregion Layers

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.conv1(x)
        x: Tensor = self.conv2(x)
        return x


class ConvHead(nn.Module):
    def __init__(self, a_in_channels, a_mid_channels, a_out_channels):
        super(ConvHead, self).__init__()

        # region Inputs
        self.in_channels: int = a_in_channels
        self.mid_channels: int = a_mid_channels
        self.out_channels: int = a_out_channels
        # endregion Inputs

        # region Layers
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=a_in_channels, out_channels=a_mid_channels, kernel_size=3,
                                          stride=2, padding=1, bias=True, groups=1)
        self.conv2: ConvDPUnit = ConvDPUnit(a_in_channels=a_mid_channels, a_out_channels=a_out_channels,
                                            a_with_bn_relu=True)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(a_mid_channels)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)
        # endregion Layers

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.conv1(x)
        x: Tensor = self.bn1(x)
        x: Tensor = self.relu1(x)
        x: Tensor = self.conv2(x)
        return x
