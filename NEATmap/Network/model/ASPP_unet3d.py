import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.conv import Conv3d
from typing import List

class ASPPConv(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int, dilation: int):
        super(ASPPConv, self).__init__(
            Conv3d(in_channel, out_channel, kernel_size=3, padding=dilation, dilation=dilation, bias=True),
            BatchNorm3d(out_channel),
            ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            Conv3d(in_channel, out_channel, kernel_size=1, bias=True),
            ReLU()
        )
    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=True)

class ASPP(nn.Module):
    def __init__(self, in_channel: int, atrous_rates: List[int], out_channel: int=32):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(Conv3d(in_channel, out_channel, kernel_size=1, bias=True),
                            BatchNorm3d(out_channel),
                            ReLU())
        ]
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channel, out_channel, rate))

        modules.append(ASPPPooling(in_channel, out_channel))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            Conv3d(len(self.convs) * out_channel, out_channel, kernel_size=1, bias=True),
            BatchNorm3d(out_channel),
            ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, 1)
        return self.project(res)

class UNetHead(nn.Sequential):
    def __init__(self, in_channel: int, num_classes: int):
        super(UNetHead, self).__init__(
            ASPP(in_channel, [12, 24, 36]),
            Conv3d(32, 32, kernel_size=3, padding=1, bias=True),
            BatchNorm3d(32),
            ReLU(),
            Conv3d(32, num_classes, kernel_size=1)
        )

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, batchnorm_flag=True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()

        self.ec1 = self.encoder(self.in_channel, 32, batchnorm=batchnorm_flag)
        self.ec2 = self.encoder(64, 64, batchnorm=batchnorm_flag)
        self.ec3 = self.encoder(128, 128, batchnorm=batchnorm_flag)
        self.ec4 = self.encoder(256, 256, batchnorm=batchnorm_flag)

        self.up3 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc3 = self.decoder(256 + 512, 256, batchnorm=batchnorm_flag)
        self.up2 = nn.Upsample(scale_factor=4., mode='trilinear', align_corners=True)
        self.dc2 = self.decoder(128 + 256, 128, batchnorm=batchnorm_flag)
        self.up1 = nn.Upsample(scale_factor=4., mode='trilinear', align_corners=True)
        self.dc1 = self.decoder(64 + 128, 64, batchnorm=batchnorm_flag)

        self.up1a = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.up2a = nn.Upsample(scale_factor=4., mode='trilinear', align_corners=True)

        self.head = UNetHead(in_channel=64, num_classes=n_classes)
        # self.dc0 = nn.Conv3d(64, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

        self.numClass = n_classes

    def encoder(self, in_channels, out_channels, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, 2*out_channels, kernel_size=1, stride=1, bias=bias),
                BatchNorm3d(2*out_channels, affine=False),
                ReLU())
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                ReLU(),
                Conv3d(out_channels, 2*out_channels, kernel_size=1, stride=1, bias=bias),
                ReLU())
        return layer


    def decoder(self, in_channels, out_channels, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU())
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                ReLU())
        return layer

    def forward(self, x):

        down1 = self.ec1(x)
        down2 = self.ec2(down1)
        down3 = self.ec3(down2)

        u3 = self.ec4(down3)
        d3 = torch.cat((self.up3(u3), down3), 1)
        u2 = self.dc3(d3)
        d2 = torch.cat((self.up2(u2), down2), 1)
        u1 = self.dc2(d2)
        d1 = torch.cat((self.up1(u1), down1), 1)
        u0 = self.dc1(d1)

        out = self.head(u0)
        out = F.interpolate(out, size=(64, 256, 256), mode='trilinear', align_corners=True)
        # out = out.view(out.numel() // self.numClass, self.numClass)
        out = self.softmax(out)

        return out

if __name__ == "__main__":

    model = UNet3D(in_channel=1, n_classes=2)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y = model(data)
    print(y.shape)