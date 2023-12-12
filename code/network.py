import torch.nn as nn
from parameters import *


class Myann(nn.Module):
    def __init__(self):
        super(Myann, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 2), stride=1)
        # self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.fc1 = nn.Linear(3776, 64)
        self.fc2 = nn.Linear(64, num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x.float()))
        # x = self.pool(x)
        x = self.relu(self.conv2(x))
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        # output = self.softmax(x)
        return output
    
ann = Myann()
