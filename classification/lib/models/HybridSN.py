import torch
from torch import nn
import torch.nn.functional as F

class HybridSN(nn.Module):
    def __init__(self, in_channel, nc, windows=17):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d( 1, 8, kernel_size = (7, 3, 3))
        self.conv2 = nn.Conv3d( 8, 16, kernel_size = (5, 3, 3))
        self.conv3 = nn.Conv3d(16, 32, kernel_size = (3, 3, 3))

        self.conv4 = nn.Conv2d(32 * (in_channel - 12), 64, (3, 3))

        self.fc1 = nn.Linear((windows - 8) * (windows - 8) * 64, 256)
        self.d1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.d2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, nc)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        n, c, d, h, w = x.shape
        x = torch.reshape(x, (n, c * d, h, w))
        x = self.conv4(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.d1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.fc3(x)
        return x
