import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No softmax (handled by loss function)

# Example: 10 input features, 16 hidden units, 2 output classes
model = MLPClassifier(input_size=16, hidden_size=16, num_classes=2)
dataset = pd.read_csv("train_dataset.csv")
features = dataset.loc[:,dataset.columns != "slice Type"]
scaler = MinMaxScaler(feature_range=(0, 1))
featuresScaled = scaler.fit_transform(features)
label = dataset.loc[:,dataset.columns == "slice Type"]
X_train, X_test, y_train, y_test = train_test_split(features,label, 
                            random_state=104,  
                            test_size=0.2,  
                            shuffle=True) 
# Convert to TensorDataset
X_Train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#label = dataset.iloc[:,-1:]

#label=np.ndarray()

#y_scaled = scaler.fit_transform(label)
#y_scaled = y_scaled.squeeze()
#print("X" , X_scaled)
#print("y" , y_scaled)
# Convert to PyTorch tensors
#X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#y_tensor = torch.tensor(label, dtype=torch.long)  # Classification -> LongTensor

# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    x,y = train_dataset
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
