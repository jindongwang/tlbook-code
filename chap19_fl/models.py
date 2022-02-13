import torch.nn as nn

class lenet5v1(nn.Module):
    def __init__(self):
        super(lenet5v1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1=nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2=nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 11)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        fealist=[]
        y = self.conv1(x)
        fealist.append(y.clone().detach())
        y=self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y=self.bn2(y)
        fealist.append(y.clone().detach())
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y,fealist

    def getfeature(self,x):
        y = self.conv1(x)
        y=self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y=self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return [y]
