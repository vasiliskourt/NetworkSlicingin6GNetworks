import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from DNNs import MLP

dataset_df = pd.read_csv("../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).to_numpy()
label = dataset_df['slice Type'].to_numpy()

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

features_tensor = torch.tensor(features, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long) - 1

k_folds_n = 5

k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

batch_size=512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fold_accuracies = []
fold_n = 0

for train_index, val_index in k_folds.split(features_tensor,label_tensor):
    fold_n += 1

    print(f"\n======= Fold {fold_n}/{k_folds_n} =======")

    X_train = features_tensor[train_index]
    y_train = label_tensor[train_index]
    X_val = features_tensor[val_index]
    y_val = label_tensor[val_index]

    train_fold_dataset = TensorDataset(X_train, y_train)
    val_fold_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(val_fold_dataset,batch_size=batch_size)

    model = MLP(input_size=16, hidden_units=32, dropout=0.3, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        
        model.train()
        avg_accuracy = 0
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            val_loss = 0
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y) 
                val_loss += loss.item() 
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
         
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

    avg_accuracy = np.mean(val_accuracies)
    fold_accuracies.append(np.mean(avg_accuracy))

    print(f"\n-> Fold {fold_n} Average Validation Accuracy: {avg_accuracy:.2f}%")

    # === Plot Loss για κάθε fold ===
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Fold {fold_n} - Train & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"MLP_K_Fold_plots/train_validation_loss_fold_{fold_n}.png")


    # === Plot Accuracy για κάθε fold ===
    plt.figure(figsize=(10, 4))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.title(f"Fold {fold_n} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"MLP_K_Fold_plots/validation_accuracy_fold_{fold_n}.png")


print("\n================================")
print(f"-> Average K-Fold Accuracy: {np.mean(fold_accuracies):.2f}%\n")

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds_n + 1), fold_accuracies, marker='o')
plt.title("Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(range(1, k_folds_n + 1))
plt.grid(True)
plt.savefig(f"MLP_K_Fold_plots/k_folds_accuracy.png")
