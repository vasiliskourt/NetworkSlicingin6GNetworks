import torch.nn as nn
import torch.nn.functional as F

# CNN Model


class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_units, dropout, num_classes):
        super(CNN1D, self).__init__()

        # Convolutional 1D
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Max Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Hidden layer
        self.fc1 = nn.Linear(input_size * 8, hidden_units)

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):

        # Convolution -> ReLU -> MaxPooling
        x = self.pool(F.relu(self.conv1(x)))

        # Flatten to 2D
        x = x.view(x.size(0), -1)

        # Fully connected layer and ReLU
        x = F.relu(self.fc1(x))

        # Dropout
        x = self.dropout(x)

        # Final Fully connected layer
        x = self.fc2(x)

        return x
