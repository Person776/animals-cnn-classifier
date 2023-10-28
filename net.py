import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, types=2):
        super(MyCNN, self).__init__()
        self.types = types

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Define the max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, self.types)  # Set the second parameter to the number of classes (types of animals)

        # Define the activation function
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply the convolutional layers and activation function
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output tensor
        x = x.view(-1, 128 * 8 * 8)

        # Apply the fully connected layers and activation function
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
