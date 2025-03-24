import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from DNNs import MLP
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


dataset_df = pd.read_csv("../train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).to_numpy()
label = dataset_df['slice Type'].to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

features_tensor = torch.tensor(features, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long) - 1

X_train, X_temp, y_train, y_temp = train_test_split(features_tensor, label_tensor, test_size=0.2, stratify=label_tensor, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size=512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=16, hidden_units=32, dropout=0.3, num_classes=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50

train_loss_l = []
train_accuracy_l = []
val_accuracy_l = []
val_loss_l = []


for epoch in range(num_epochs):

    model.train()
    train_loss = 0
    train_correct_prediction = 0
    train_total_checked = 0

    for batch_X, batch_y in train_loader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct_prediction += (predicted == batch_y).sum().item()
        train_total_checked += batch_y.size(0)

    avg_train_loss  = train_loss / len(train_loader)
    train_accuracy  = 100 * train_correct_prediction / train_total_checked      
   

    model.eval()
    val_correct_prediction = 0
    val_total_checked = 0
    val_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            val_loss += loss.item() 
            _, predicted = torch.max(outputs, 1)
            val_correct_prediction += (predicted == batch_y).sum().item()
            val_total_checked += batch_y.size(0)

    val_accuracy  = 100 * val_correct_prediction / val_total_checked
    avg_val_loss = val_loss / len(val_loader)


    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.5f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    
    train_loss_l.append(avg_train_loss)
    train_accuracy_l.append(train_accuracy)
    val_accuracy_l.append(val_accuracy)
    val_loss_l.append(avg_val_loss)

torch.save(model.state_dict(), "MLP_model/mlp_state.pth")

plt.figure(figsize=(10, 4))
plt.plot(train_loss_l, label='Train Loss')
plt.plot(val_loss_l, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("MLP_train_plots/train_validation_loss.png")


plt.figure(figsize=(10, 4))
plt.plot(train_accuracy_l, label='Train Accuracy')
plt.plot(val_accuracy_l, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("MLP_train_plots/train_validation_accuracy.png")


model = MLP(input_size=16, hidden_units=32, dropout=0.3, num_classes=3)
model.load_state_dict(torch.load("MLP_model/mlp_state.pth"))
model.to(device)
model.eval()

test_prediction = []
test_correct = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)

        _, predicted = torch.max(outputs, 1)
        test_prediction.extend(predicted.cpu().numpy())
        test_correct.extend(batch_y.numpy())

test_accuracy = accuracy_score(test_correct, test_prediction) * 100
print(f"-> Test Accuracy: {test_accuracy:.2f}%")
