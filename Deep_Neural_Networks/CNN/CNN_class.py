import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_units, dropout, num_classes):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        self.fc1 = nn.Linear(input_size * 8, hidden_units)  
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x)) 
        x= self.dropout(x)
        x = self.fc2(x)  
        return x