import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, 5, padding=2)
        self.conv3 = nn.Conv2d(12, 3, 3, padding=1)

        # self.conv1 = nn.Conv2d(3, 12, 5, padding=2)
        # self.conv2 = nn.Conv2d(12, 12, 7, padding=3)
        # self.conv3 = nn.Conv2d(12, 3, 5, padding=2)


        # self.pool1 = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.pool1(x))
        return x