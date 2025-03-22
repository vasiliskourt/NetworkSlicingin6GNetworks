from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from DNNs import CNN1D
from sklearn.model_selection import KFold, train_test_split

# K-Fold Cross Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
dataset_df = pd.read_csv("../train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).to_numpy()
label = dataset_df['slice Type'].to_numpy() - 1

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

features_tensor = torch.tensor(features, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long) 

X_train, X_test, y_train, y_test = train_test_split(features_tensor, label_tensor, test_size=0.2, random_state=42, stratify=label)

# Reshape for Conv1D
X_train = X_train.reshape(-1, 1, 16)
X_test = X_test.reshape(-1, 1, 16)
#X_train = X_train.unsqueeze(1)  # Now shape is (batch_size, 1, 16)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(X_train.shape,y_train.shape)
model = CNN1D(num_classes=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
cv_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset,batch_size=batch_size,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    for epoch in range(num_epochs):

        model.train()

        total_loss = 0
        correct_prediction = 0
        total_checked = 0

        for batch_X, batch_y in train_loader:

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_prediction += (predicted == batch_y).sum().item()
            total_checked += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        #accuracy = 100 * correct_prediction / total_checked      
        acc = correct_prediction / total_checked
        print(f"Validation Accuracy: {acc:.4f}")
        cv_accuracies.append(acc)
        #model.eval()
        
        '''
        with torch.no_grad():

            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                correct_prediction += (predicted == batch_y).sum().item()
                total_checked += batch_y.size(0)
                val_total += batch_y.size(0)

        accuracy2 = 100 * correct_prediction / total_checked

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {accuracy: .2f}% | "
            f"Val Acc: {accuracy2:.2f}%", flush=True)
'''     
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
correct_prediction = 0
total_checked = 0
val_total = 0
with torch.no_grad():
     for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                correct_prediction += (predicted == batch_y).sum().item()
                total_checked += batch_y.size(0)
                val_total += batch_y.size(0)

test_accuracy = correct_prediction/total_checked
print(f"\nAverage CV Accuracy: {np.mean(cv_accuracies):.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
plt.figure(figsize=(10, 4))
plt.plot(total_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)
#plt.show()

plt.figure(figsize=(10, 4))
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.legend()
plt.grid(True)
#plt.show()

model = torch.load("mlp.pth")
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
print(f"Test Accuracy: {test_accuracy:.2f}%")

confusion_matrix_ = confusion_matrix(test_correct, test_prediction)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_, annot=True, fmt='d', cmap='Blues', xticklabels=["Slice type 1", "Slice type 2", "Slice type 3"], yticklabels=["Slice type 1", "Slice type 2", "Slice type 3"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Test Set")