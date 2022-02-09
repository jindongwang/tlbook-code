import torch
import torch.nn as nn


class TNNAR(nn.Module):
    def __init__(self, n_class=19):
        super(TNNAR, self).__init__()
        self.n_class = n_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 2, 100),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, self.n_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.reshape(-1, 32 * 2)
        x = self.fc1(x)
        fea = x
        out = self.fc2(x)

        return fea, out

    def predict(self, x):
        _, out = self.forward(x)
        return out


if __name__ == '__main__':
    a = torch.randn(2, 9, 9, 1).cuda()
    net = TNNAR(2).cuda()
    out = net(a)
    print(out)
