import torch
from torch import nn
import torch.nn.functional as F

'''Implementation for CONVOLUTIONAL NEURAL NETWORKS FOR HYPERSPECTRAL IMAGE CLASSIFICATION'''

class CNN_HSI(nn.Module):
    def __init__(self, in_channel, nc, windows=5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, nc, 1),
            nn.ReLU(),
            nn.AvgPool2d(windows, 1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = torch.squeeze(out3)
        return out4


if __name__ == '__main__':
    net = CNN_HSI(128, 19, 21)
    input = torch.rand((256, 128, 21, 21))
    out = net(input)
    print(out.shape)
