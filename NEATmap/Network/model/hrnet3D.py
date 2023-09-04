import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.container import ModuleList

BN_MOMENTUM = 0.2


class PlaceHolder(nn.Module):

    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, inputs):
        return inputs

class HRNetConv3x3x3(nn.Module):

    def __init__(self, inchannels, outchannels, stride=1, padding=0):
        super(HRNetConv3x3x3, self).__init__()

        self.conv = nn.Conv3d(inchannels, outchannels, kernel_size=3, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm3d(outchannels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x

# Stem
class HRNetStem(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(HRNetStem, self).__init__()

        # resolution to 1/2
        self.conv1 = HRNetConv3x3x3(inchannels, outchannels, stride=2, padding=1)
        # resolution to 1/4
        self.conv2 = HRNetConv3x3x3(outchannels, outchannels, stride=2, padding=1)

    def forward(self, inputs):
        
        x = self.conv1(inputs) # 1/2
        x = self.conv2(x)      # 1/4

        return x 
    
class HRNetInput(nn.Module):

    def __init__(self, inchannels, outchannels, stage1_inchannels):
        super(HRNetInput, self).__init__()

        self.stem = HRNetStem(inchannels, outchannels)

        self.in_change_conv = nn.Conv3d(outchannels, stage1_inchannels, kernel_size=1, stride=1, bias=False)

        self.in_change_bn   = nn.BatchNorm3d(stage1_inchannels, momentum=BN_MOMENTUM)
        self.relu           = nn.ReLU()

    def forward(self, inputs):
        
        x = self.stem(inputs) # outchannels = 64
        x = self.in_change_conv(x) # stage1_inchannels = 32
        x = self.in_change_bn(x)
        x = self.relu(x)

        return x

class NormalBlock(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(NormalBlock, self).__init__()

        self.conv1 = HRNetConv3x3x3(inchannels=inchannels, outchannels=outchannels, stride=1, padding=1)

        self.conv2 = HRNetConv3x3x3(inchannels=outchannels, outchannels=outchannels, stride=1, padding=1)

    def forward(self, inputs):

         x = self.conv1(inputs)
         x = self.conv2(x)

         return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ResidualBlock, self).__init__()

        self.conv1 = HRNetConv3x3x3(inchannels=inchannels, outchannels=outchannels, stride=1, padding=1)

        self.conv2 = nn.Conv3d(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm3d(outchannels, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU()

    def forward(self, inputs):
        residual = inputs

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn2(x)
        x = residual + x
        x = self.relu(x)

        return x

# Sequential
# LayerList  -- layer[0](squential), layer[1](squential)

class HRNetStage(nn.Module):

    def __init__(self, stage_channels, block):
        super(HRNetStage, self).__init__()
        # stage_channels is list [32, 64 ,128]

        self.stage_channels = stage_channels
        self.stage_branch_num = len(stage_channels)
        self.block = block
        self.block_num = 4

        self.stage_layers = self.create_stage_layers()

    def forward(self, inputs):
        outs = []
        
        for i in range(self.stage_branch_num):
            x = inputs[i]
            out = self.stage_layers[i](x)
            outs.append(out)

        return outs

    def create_stage_layers(self):
        tostage_layers = [] 

        for i in range(self.stage_branch_num):
            branch_layer = [] 
            for j in range(self.block_num):
                branch_layer.append(self.block(self.stage_channels[i], self.stage_channels[i]))
            branch_layer = nn.Sequential(*branch_layer) # block1,block2
            tostage_layers.append(branch_layer)

        return nn.ModuleList(tostage_layers)

class HRNetTrans(nn.Module):

    def __init__(self,  old_branch_channels, new_branch_channels):
        super(HRNetTrans, self).__init__()

        # old_branch_channnels is list [branch1, branch2]
        # len(old_branch_channnels) = branch_num

        # stages' channels
        self.old_branch_channels = old_branch_channels
        self.new_branch_channels = new_branch_channels

        # branch number
        self.old_branch_num = len(old_branch_channels)
        self.new_branch_num = len(new_branch_channels)

        self.trans_layers = self.create_new_branch_trans_layers()

    def forward(self, inputs):
        
        # outputs list
        outs = []
        for i in range(self.old_branch_num):
            x = inputs[i]
            out = []

        
            for j in range(self.new_branch_num):
                y = self.trans_layers[i][j](x)
                out.append(y)

            if len(outs) == 0:
                outs = out
            else:
                for i in range(self.new_branch_num):
                    outs[i] = outs[i] + out[i]

        return outs

    def create_new_branch_trans_layers(self):
        #layerlist
        totran_layers = [] 

        for i in range(self.old_branch_num):
            branch_trans = []
            for j in range(self.new_branch_num):
                if i == j:
                    layer = PlaceHolder()
                elif i < j:
                    layer = []
                    inchannels = self.old_branch_channels[i]
                     # j -i > 0
                     # 1 --> downsample
                    for k in range(j - i):
                        layer.append(nn.Conv3d(in_channels=inchannels, out_channels=self.new_branch_channels[j], 
                                                kernel_size=1, bias=False))
                        layer.append(nn.BatchNorm3d(self.new_branch_channels[j], momentum=BN_MOMENTUM))
                        layer.append(nn.ReLU())
                        # 下采样率: 1/2
                        layer.append(nn.Conv3d(in_channels=self.new_branch_channels[j], out_channels=self.new_branch_channels[j], 
                                                kernel_size=3, stride=2, padding=1, bias=False))
                        layer.append(nn.BatchNorm3d(self.new_branch_channels[j], momentum=BN_MOMENTUM))
                        layer.append(nn.ReLU())
                        inchannels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                elif i > j:
                    layer = []
                    inchannels = self.old_branch_channels[i]
                    for k in range(i - j):
                        layer.append(nn.Conv3d(in_channels=inchannels, out_channels=self.new_branch_channels[j], 
                                                kernel_size=1, bias=False))
                        layer.append(nn.BatchNorm3d(self.new_branch_channels[j], momentum=BN_MOMENTUM))
                        layer.append(nn.ReLU())
                        layer.append(nn.Upsample(scale_factor=2.))
                        inchannels = self.new_branch_channels[j]

                    layer = nn.Sequential(*layer)
                branch_trans.append(layer)

            branch_trans = nn.ModuleList(branch_trans)
            totran_layers.append(branch_trans)

        return nn.ModuleList(totran_layers)

Fusion_Mode = ['keep', 'fuse', 'multi']

class HRNetFusion(nn.Module):

    def __init__(self, stage4_channels, mode='keep'):
        super(HRNetFusion, self).__init__()

        assert mode in Fusion_Mode, \
            'Please inout mode is [keep, fuse, multi], in HRNetFusion'

        self.stage4_channels = stage4_channels
        self.mode = mode

        # 根据模式进行构建融合层
        self.fuse_layer = self.create_fuse_layers()

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        outs = []

        if self.mode == Fusion_Mode[0]:
            out = self.fuse_layer(x1)
            outs.append(out)
        elif self.mode == Fusion_Mode[1]:
            out = self.fuse_layer[0](x1)
            out = out + self.fuse_layer[1](x2)
            out = out + self.fuse_layer[2](x3)
            out = out + self.fuse_layer[3](x4)
            outs.append(out)
        elif self.mode == Fusion_Mode[2]:
            out1 = self.fuse_layer[0][0](x1)
            out1 = out1 + self.fuse_layer[0][1](x2)
            out1 = out1 + self.fuse_layer[0][2](x3)
            out1 = out1 + self.fuse_layer[0][3](x4)
            outs.append(out1)

            out2 = self.fuse_layer[1](out1)
            outs.append(out2)

            out3 = self.fuse_layer[2](out2)
            outs.append(out3)

            out4 = self.fuse_layer[3](out3)
            outs.append(out4)
        
        return outs

    def create_fuse_layers(self):

        layer = None

        if self.mode == 'keep':
            layer = self.create_keep_fusion_layers()
        elif self.mode == 'fuse':
            layer = self.create_fuse_fusion_layers()
        elif self.mode == 'multi':
            layer = self.create_multi_fusion_layers()

        return layer

    def create_keep_fusion_layers(self):
        self.outchannels = self.stage4_channels[0]
        return PlaceHolder()
    
    def create_fuse_fusion_layers(self):
        layers = []

        outchannel = self.stage4_channels[3] # outchannels

        for i in range(len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer = []

            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv3d(in_channels=inchannel, out_channels=outchannel,
                                        kernel_size=1, bias=False))
                layer.append(nn.BatchNorm3d(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))

            layer = nn.Sequential(*layer)
            layers.append(layer)

        self.outchannels = outchannel
        return nn.ModuleList(layers)    

    def create_multi_fusion_layers(self):
        multi_fuse_layers = []

        layers = []

        outchannel = self.stage4_channels[3] # outchannels

        for i in range(len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer = []

            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv3d(in_channels=inchannel, out_channels=outchannel,
                                        kernel_size=1, bias=False))
                layer.append(nn.BatchNorm3d(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))
                
            layer = nn.Sequential(*layer)
            layers.append(layer)

        # 第一个fuse - layer
        multi_fuse_layers.append(nn.ModuleList(layers))

        # branch1 to branch2
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        # branch2 to branch3
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        # branch3 to branch4
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        self.outchannels = outchannel
        return nn.ModuleList(multi_fuse_layers)

class HRNetOutPut(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(HRNetOutPut, self).__init__()

        self.conv = nn.Conv3d(inchannels, outchannels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm3d(outchannels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        # self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, inputs):

        N = len(inputs)

        outs = []
        for i in range(N):

            out = self.conv(inputs[i])
            out = self.bn(out)
            # out = self.avg_pool(out)
            out = self.relu(out)
            
            outs.append(out)

        return outs

class HRNetClassification(nn.Module):

    def __init__(self, num_classes):
        super(HRNetClassification, self).__init__()

        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        outs = []

        for i in range(len(inputs)):
            out = self.flatten(inputs[i])
            out = self.fc_out(out)
            out = self.sigmoid(out)
            outs.append(out)

        return outs

class HRNet(nn.Module):
    
    def __init__(self, num_classes = 2, width = 64):
        super(HRNet, self).__init__()

        self.width = width

        if self.width == 16:
            self.stage_channels = [[16], [16, 32], [16, 32, 64], [16, 32, 64, 128]]
        elif self.width == 32:
            self.stage_channels = [[32], [32, 64], [32, 64, 128], [32, 64, 128, 256]]
        elif self.width == 64:
            self.stage_channels = [[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]
        elif self.width == 128:
            self.stage_channels = [[128], [128, 256], [128, 256, 512], [128, 256, 512, 1024]]

        
        self.input  = HRNetInput(1, outchannels=64, stage1_inchannels=self.width)

        # inputs_channels: [32, 64, 128]
        # outputs 3 unit
        self.stage1 = HRNetStage(self.stage_channels[0], ResidualBlock)
        self.trans1 = HRNetTrans(self.stage_channels[0], self.stage_channels[1])

        self.stage2 = HRNetStage(self.stage_channels[1], ResidualBlock)
        self.trans2 = HRNetTrans(self.stage_channels[1], self.stage_channels[2])

        self.stage3 = HRNetStage(self.stage_channels[2], ResidualBlock)
        self.trans3 = HRNetTrans(self.stage_channels[2], self.stage_channels[3])

        self.stage4 = HRNetStage(self.stage_channels[3], ResidualBlock)
        self.fuse_layer = HRNetFusion(self.stage_channels[3], mode='fuse')

        self.last_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=self.stage_channels[3][3]*4,
                out_channels=self.stage_channels[3][3]*4,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm3d(self.stage_channels[3][3]*4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=self.stage_channels[3][3]*4,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0),
        )

        self.keep_last_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm3d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0),
        )
        self.fuse_last_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm3d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=512,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0),
        )
        self.softmax = nn.Softmax(dim=1).cuda()
        # self.output = HRNetOutPut(self.fuse_layer.outchannels, 256)

        # self.classifier = HRNetClassification(num_classes=num_classes)

    def forward(self, inputs):

        x = self.input(inputs)
        x = [x]

        x = self.stage1(x)
        x = self.trans1(x) #  32, 64, 128, 256

        x = self.stage2(x)
        x = self.trans2(x) #  32, 64, 128, 256

        x = self.stage3(x)
        x = self.trans3(x) #  32, 64, 128, 256

        x = self.stage4(x) #  32, 64, 128, 256
        x = self.fuse_layer(x) # keep 32, 64*64

        # Upsampling
        # x0_s, x0_h, x0_w = x[0].size(2), x[0].size(3), x[0].size(4)
        # x1 = F.interpolate(x[1], size=(x0_s, x0_h, x0_w), mode='trilinear', align_corners=True)#x1 = F.upsample(x[1], size=(x0_s, x0_h, x0_w), mode='trilinear')
        # x2 = F.interpolate(x[2], size=(x0_s, x0_h, x0_w), mode='trilinear', align_corners=True)
        # x3 = F.interpolate(x[3], size=(x0_s, x0_h, x0_w), mode='trilinear', align_corners=True)
        # # batch_size = x[0].size(0)
        # x = torch.cat([x[0], x1, x2, x3], 1)
        # # x = x[0].view(batch_size, 8, 64, 256, 256)
        # x = self.last_layer(x)
        # # x = self.keep_last_layer(x[0])
        x = self.fuse_last_layer(x[0])
        x = F.interpolate(x, size=(64, 256, 256), mode='trilinear', align_corners=True)
        x = self.softmax(x)
        # x = self.output(x)
        # x = self.classifier(x)

        return x 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":

    model = HRNet(width=64)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y = model(data)

    # for i in range(len(y)):
    #     print(y)
    print(y.shape)
