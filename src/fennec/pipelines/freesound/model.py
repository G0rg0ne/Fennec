import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CNNAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Args:
            input_size (tuple): Tuple of (time_steps, n_mfcc), e.g., (32, 12).
            num_classes (int): Number of output classes for classification.
        """
        super(CNNAudioClassifier, self).__init__()
        self.input_size = input_size  # (time_steps, n_mfcc)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Dynamically calculate the flattened size
        self.flatten_size = self._get_flatten_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_flatten_size(self, input_size):
        """
        Calculate the size of the flattened tensor after convolution and pooling.
        """
        x = torch.zeros(1, 1, *input_size)  # Dummy input: (batch_size, channels, time_steps, n_mfcc)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        return x.numel()  # Total number of elements in the tensor

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv2D
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x