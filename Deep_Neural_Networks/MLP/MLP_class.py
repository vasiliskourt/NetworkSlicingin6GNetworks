import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, dropout, num_classes):
        super(MLP, self).__init__()
        
        # Hidden layer
        self.fc1 = nn.Linear(input_size, hidden_units)
        
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Output layer        
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        # Fully connected layer and ReLU
        x = F.relu(self.fc1(x))
        
        # Dropout
        x = self.dropout(x)

        # Final Fully connected layer
        x = self.fc2(x)
        
        return x