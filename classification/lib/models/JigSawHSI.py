from torch import nn
import torch


class JigSawHSI(nn.Module):
    def __init__(self, inchannel, n_classes, windows=27):
        super().__init__()
        self.norm = nn.LayerNorm(inchannel)
        self.conv1 = self.build_conv3d(1, 64, kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.conv2 = self.build_conv3d(64, 32, kernel_size=(1, 1, 5), stride=(1, 1, 2))
        self.conv3 = self.build_conv3d(32, 16, kernel_size=(1, 1, 3), stride=(1, 1, 2))

        self.jigsaw_stages = nn.ModuleList()
        for i in range(4):
            if i == 0:
                channel = 64
            else:
                channel = 16
            self.jigsaw_stages.append(self.build_jigsaw_stage(224, channel, 2 * i + 3))

        self.dense1 = nn.Linear(112 * windows * windows, 256)
        self.dropout1 = nn.Dropout(0.4)

        self.dense2 = nn.Linear(inchannel, 16)
        self.dropout2 = nn.Dropout(0.4)
        self.dense3 = nn.Linear(16, 16)
        self.dropout3 = nn.Dropout(0.4)

        self.dense4 = nn.Linear(272, 128)
        self.dropput4 = nn.Dropout(0.4)
        self.dense5 = nn.Linear(128, n_classes)
    
    @staticmethod
    def build_conv3d(inchannel, channel, kernel_size, stride):
        return nn.Sequential(
            nn.Conv3d(inchannel, channel, kernel_size, stride),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def build_jigsaw_stage(inchannel, channel, kernel_size: int):
        return nn.Sequential(
            nn.Conv2d(inchannel, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16, 16, kernel_size, padding=(kernel_size - 1)//2, stride=1),
            nn.Conv2d(16, channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        sz = x.shape[-1]
        y = x[:, :, (sz - 1) // 2, (sz - 1) // 2].clone()

        # x <- [n,c,w,w]
        x = x.permute(0, 2, 3, 1)
        # x <- [n,w,w,c]
        x = self.norm(x)
        x = x.unsqueeze(1)
        # x <- [n,1,w,w,c]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x <- [n,16,w,w,x]
        b, _, h, w, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape((b, -1, h, w))
        # x <- [n,x*16,w,w]

        outputs = []
        for stage in self.jigsaw_stages:
            r = stage(x)
            outputs.append(r)
        x = torch.cat(outputs, dim=1)
        x = x.flatten(1)
        x = self.dropout1(self.dense1(x))

        y = self.dropout2(self.dense2(y))
        y = self.dropout3(self.dense3(y))

        o = torch.cat([x, y], dim=1)
        o = self.dropput4(self.dense4(o))
        return self.dense5(o)

if __name__ == '__main__':
    m = JigSawHSI(128, 19, 27)
    input = torch.rand(256, 128, 27, 27)
    o = m(input)
    print(o.shape)
