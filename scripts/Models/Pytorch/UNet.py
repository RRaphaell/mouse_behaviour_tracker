import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
        Double Convolution layer with both 2 BN and Activation Layer in between
        Conv2d==>BN==>Activation==>Conv2d==>BN==>Activation
    """

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    """
      A maxpool layer followed by a Double Convolution.
      MaxPool2d==>double_conv.
    """

    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.activation = nn.ReLU(inplace=True)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.activation(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # DownSampling Block
        self.down_block1 = DoubleConv(in_channel, 16)
        self.down_block2 = DownConv(16, 32)
        self.down_block3 = DownConv(32, 64)
        self.down_block4 = DownConv(64, 128)
        self.down_block5 = DownConv(128, 256)

        # # UpSampling Block
        self.up_block1 = UpSample(256, 128)
        self.up_block2 = UpSample(128, 64)
        self.up_block3 = UpSample(64, 32)
        self.up_block4 = UpSample(32, 16)
        self.up_block5 = nn.Conv2d(16, out_channel, 1)

    def forward(self, x):
        # Down
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        # # Up
        x6 = self.up_block1(x5, x4)
        x7 = self.up_block2(x6, x3)
        x8 = self.up_block3(x7, x2)
        x9 = self.up_block4(x8, x1)
        x10 = self.up_block5(x9)
        return x10
