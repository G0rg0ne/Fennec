import torch
from torch import nn
import torch.nn.functional as F


    
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, num_classes)  # Fully connected output layer
    
    def forward(self, x):
        # LSTM expects input shape (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Get the output of the last time step
        lstm_last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Pass through the fully connected layer
        output = self.fc(lstm_last_output)  # (batch_size, num_classes)
        return output