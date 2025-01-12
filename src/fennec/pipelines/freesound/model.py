import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CNNAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, initial_temperature=1.0):
        """
        Args:
            input_size (tuple): Tuple of (time_steps, n_mfcc), e.g., (32, 12).
            num_classes (int): Number of output classes for classification.
            initial_temperature (float): Initial value for the temperature parameter.
        """
        super(CNNAudioClassifier, self).__init__()
        self.input_size = input_size  # (time_steps, n_mfcc)

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)  # Output layer: num_classes units

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Add channel dimension for Conv2D
        x = x.unsqueeze(1)  # Input: (batch_size, 1, time_steps, n_mfcc)

        # First block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Second block
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Third block
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Fourth block with global average pooling
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)  # Output: (batch_size, 128, 1, 1)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Output: (batch_size, 128)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # Output logits (no activation here)

        # Apply temperature scaling to logits
        scaled_logits = logits / self.temperature

        return scaled_logits