# REFERENCES
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # convolutional layer 1: 3x224x224 -> 16x224x224
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # convolutional layer 2: 16x224x224 -> 16x112x112
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # convolutional layer 3: 16x112x112 -> 8x112x112
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        # convolutional layer 4: 8x112x112 -> 8x56x56 (after max pool)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)

        # fully connected layer: flattened 8x56x56 -> 32
        self.fc1 = nn.Linear(8 * 56 * 56, 32)
        self.fc2 = nn.Linear(32, 7)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # flatten data
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        # pass through fully connected layer
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        # final output
        x = F.softmax(self.fc2(x), dim=1)

        return x
