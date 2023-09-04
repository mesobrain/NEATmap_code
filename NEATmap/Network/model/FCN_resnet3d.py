import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Resnet3d(nn.Module):
    
    def __init__(self, block, num_block):
        super(Resnet3d, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv3d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_block[0])
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channel*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))

        self.in_channel = channel * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

def resnet18_3d():
    return Resnet3d(BasicBlock, [2, 2, 2, 2])

def resnet34_3d():
    return Resnet3d(BasicBlock, [3, 4, 6, 3])

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(),
            nn.Conv3d(inter_channels, channels, 1),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )

class FCN_resnet(nn.Module):
    def __init__(self, backbone, classifer):
        super(FCN_resnet, self).__init__()
        self.backbone = backbone
        self.classifer = classifer
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        input_shape = x.shape[-3:]
        feature = self.backbone(x)

        x = self.classifer(feature)
        result = F.interpolate(x, size=input_shape, mode='trilinear', align_corners=True)
        result = self.softmax(result)

        return result

def fcn_resnet34(num_classes=2):

    backbone = resnet34_3d()
    out_inplanes = 512
    
    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN_resnet(backbone, classifier)

    return model

if __name__ == "__main__":

    model = fcn_resnet34(num_classes=2)

    import numpy as np
    
    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y = model(data)
    print(y.shape)