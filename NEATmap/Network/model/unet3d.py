import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.conv import Conv3d

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
        self.up2 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc2 = self.decoder(128 + 256, 128, batchnorm=batchnorm_flag)
        self.up1 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc1 = self.last_decoder(64 + 128, 64, n_classes, batchnorm=batchnorm_flag)

        # self.dc0 = nn.Conv3d(64, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

        self.numClass = n_classes

    def encoder(self, in_channels, out_channels, bias=False, batchnorm=False):
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


    def decoder(self, in_channels, out_channels, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU())
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                ReLU())
        return layer

    def last_decoder(self, in_channels, out_channels, num_classes, batchnorm):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
            )
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                ReLU(),
                Conv3d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
            )
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
        out = self.dc1(d1)

        out_map = F.interpolate(out, size=(64, 256, 256), mode='trilinear', align_corners=True)
        # out = out_map.view(out_map.numel() // self.numClass, self.numClass)
        out = self.softmax(out_map)
        # out = self.softmax(out)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
if __name__ == "__main__":

    model = UNet3D(in_channel=1, n_classes=2)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y = model(data)
    print(y.shape)