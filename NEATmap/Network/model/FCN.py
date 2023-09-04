import torch
import torch.nn as nn

class Conv3x3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias) -> None:
        super(Conv3x3x3, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, input):

        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)

        return x

class FCN(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        self.in_channels = in_channels
        self.num_classes = num_classes
        super(FCN, self).__init__()

        filters = [5, 10, 20, 50]
        self.conv1 = Conv3x3x3(self.in_channels, filters[0], kernel_size=3, stride=1, padding=1, bias=False)#2->5
        self.conv2 = Conv3x3x3(filters[0], filters[1], kernel_size=5, stride=1, padding=2, bias=False)#5->10
        self.conv3 = Conv3x3x3(filters[1], filters[2], kernel_size=5, stride=1, padding=2, bias=False)#10->20
        self.conv4 = Conv3x3x3(filters[2], filters[3], kernel_size=3, stride=1, padding=1, bias=False)#20->50
        self.last_layer = self.last_conv(filters[3], self.num_classes)

        self.softmax = nn.Softmax(dim=1)

    def last_conv(self, in_channels, num_classes, kernel_size=1, stride=1, padding=0):
        
        layer = nn.Sequential(
            nn.Conv3d(in_channels, num_classes, kernel_size, stride, padding),
            nn.BatchNorm3d(num_classes),
            nn.ReLU()
        )

        return layer

    def forward(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.last_layer(out)

        final_out = self.softmax(out)

        return final_out

if __name__ == "__main__":

    model = FCN(in_channels=1, num_classes=2)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y = model(data)
    print(y.shape)