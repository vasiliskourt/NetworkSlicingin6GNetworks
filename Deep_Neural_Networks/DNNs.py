import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, dropout, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CNN1D(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: (32, 16)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output: (64, 16)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (64, 8)
        
        self.fc1 = nn.Linear(64 * 4, 128)  # Fully Connected Layer
        self.fc2 = nn.Linear(128, num_classes)  # Output Layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # Fully Connected Layer
        x = self.fc2(x)  # Output Layer
        return x
'''
class CNN1D(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # (batch, 32, 16)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (batch, 64, 16)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)  # (batch, 64, 14)

        self.fc1 = nn.Linear(64 * 14, 128)  # Fixed: Correct flattened size
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 15)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 14)
        x = x.view(x.size(0), -1)  # Flatten (batch, 64 * 14)
        x = F.relu(self.fc1(x))  # Fully Connected Layer
        x = self.fc2(x)  # Output Layer
        return x
'''