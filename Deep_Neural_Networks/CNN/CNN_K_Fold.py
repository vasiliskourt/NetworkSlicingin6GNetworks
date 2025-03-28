import torch
import torch.nn as nn
import torch.optim as optim
from CNN_class import CNN1D
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import time

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

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

val_fold_accuracies = []
train_fold_accuracies = []
train_time_l = []
train_loss_l = []
val_loss_l = []
fold_n = 0

for train_index, val_index in k_folds.split(features_tensor,label_tensor):
    fold_n += 1

    print(f"\n======= Fold {fold_n}/{k_folds_n} =======")

    X_train = features_tensor[train_index]
    y_train = label_tensor[train_index]
    X_val = features_tensor[val_index]
    y_val = label_tensor[val_index]
    
    X_train = X_train.reshape(-1, 1, 16)
    X_val = X_val.reshape(-1,1,16)

    train_fold_dataset = TensorDataset(X_train, y_train)
    val_fold_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(val_fold_dataset,batch_size=batch_size)

    model = CNN1D(input_size=16, hidden_units=32, dropout=0.3, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 40

    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    train_time_start = time.time()

    for epoch in range(num_epochs):
        
        model.train()
        val_avg_accuracy = 0
        train_avg_accuracy = 0
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

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy  = 100 * train_correct_prediction / train_total_checked      


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

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Train Acc: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f}%")
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    train_time_end = time.time()
    training_time = train_time_end - train_time_start
    train_time_l.append(training_time)
    val_avg_accuracy = np.mean(val_accuracies)
    train_avg_accuracy = np.mean(train_accuracies)
    val_fold_accuracies.append(np.mean(val_avg_accuracy))
    train_fold_accuracies.append(np.mean(train_avg_accuracy))
    train_loss_l.append(np.mean(train_losses))
    val_loss_l.append(np.mean(val_losses))

    print(f"\n-> Fold {fold_n} Average Validation Accuracy: {val_avg_accuracy:.2f}%, Average Train Accuracy: {train_avg_accuracy:.2f}% , Training Time: {training_time:.3f} seconds")

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"(CNN) Fold {fold_n} - Train & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"CNN_K_Fold_plots/train_validation_loss_fold_{fold_n}.png")

    plt.figure(figsize=(10, 4))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.title(f"(CNN) Fold {fold_n} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"CNN_K_Fold_plots/train_validation_accuracy_fold_{fold_n}.png")

print("================================\n")
print(f"-> Average K-Fold Train Accuracy: {np.mean(train_fold_accuracies):.2f}%\n")
print(f"-> Average K-Fold Validation Accuracy: {np.mean(val_fold_accuracies):.2f}%\n")
print(f"-> Average CNN Training Time: {np.mean(train_time_l):.2f} seconds\n")
print("================================\n")

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds_n + 1), val_fold_accuracies, label="Val Accuracy")
plt.plot(range(1, k_folds_n + 1), train_fold_accuracies, label="Train Accuracy")
plt.title("(CNN) Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(range(1, k_folds_n + 1))
plt.legend()
plt.grid(True)
plt.savefig(f"CNN_K_Fold_plots/k_folds_accuracy.png")

plt.figure(figsize=(10, 4))
plt.plot(range(1, k_folds_n + 1), train_time_l, label="Time")
plt.title("(CNN) Time to train")
plt.xlabel("Epoch")
plt.ylabel("Training Time")
plt.legend()
plt.grid(True)
plt.savefig(f"CNN_K_Fold_plots/training_time.png")

with open("CNN_report/CNN_report.txt", "w") as file:
    file.write("---------CNN Report---------\n")
    file.write(f"-> Epochs: {num_epochs}\n")
    file.write("\n-> Validation Accuracy:\n")
    for i, acc in enumerate(val_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\n-> Average Validation Accuracy: {np.mean(val_fold_accuracies):.2f}%\n")
    file.write("\n-> Train Accuracy:\n")
    for i, acc in enumerate(train_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\n-> Average Train Accuracy: {np.mean(train_fold_accuracies):.2f}%\n")
    file.write(f"\n-> Train Time:\n")
    for i, times in enumerate(train_time_l):
        file.write(f"Fold {i+1}: {times:.3f} seconds\n")
    file.write(f"\n-> Average Train Time: {np.mean(train_time_l):.3f} seconds\n")
    file.write("\n-> Train Loss: (Avg per fold)\n")
    for i, loss in enumerate(train_loss_l):
        file.write(f"Fold {i+1}: {loss:.3f}\n")
    file.write("\n-> Validation Loss (Avg per fold):\n")
    for i, loss in enumerate(val_loss_l):
        file.write(f"Fold {i+1}: {loss:.3f}\n")

print("Report saved:\n -> CNN/CNN_report\n")
print("Plots saved:\n -> CNN/CNN_K_Fold_plots\n")
print("================================\n")

