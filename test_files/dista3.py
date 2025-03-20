import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 📌 1. Φόρτωση του dataset
df = pd.read_csv("train_dataset.csv")

# 📌 2. Διαχωρισμός χαρακτηριστικών και labels
X = df.drop(columns=['slice Type']).values
y = df['slice Type'].values  # Κατηγορίες (1, 2, 3)
y = y - 1  # 🔹 Διόρθωση: μετατροπή labels ώστε να ξεκινούν από 0

# 📌 3. Κανονικοποίηση των χαρακτηριστικών (για MLP & XGBoost)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 4. Διαχωρισμός σε train και test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 📌 5. Εκπαίδευση Random Forest Classifier
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# 📌 6. Εκπαίδευση XGBoost Classifier
start_time = time.time()
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

# 📌 7. Εκπαίδευση MLP (Deep Learning)
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

# 🔹 Ορισμός hyperparameters για το MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = MLP(input_size=16, hidden_units=64, dropout=0.3, num_classes=3).to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 🔹 Μετατροπή δεδομένων σε PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 🔹 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 🔹 Εκπαίδευση MLP (10 Epochs)
num_epochs = 10
start_time = time.time()

for epoch in range(num_epochs):
    mlp_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

mlp_time = time.time() - start_time

# 🔹 Αξιολόγηση MLP
mlp_model.eval()
with torch.no_grad():
    outputs = mlp_model(X_test_tensor)
    _, mlp_preds = torch.max(outputs, 1)

mlp_acc = accuracy_score(y_test, mlp_preds.cpu().numpy())

# 📌 8. Εκτύπωση Αποτελεσμάτων
print(f"🎯 Random Forest Accuracy: {rf_acc:.4f} (Training Time: {rf_time:.2f}s)")
print(f"🎯 XGBoost Accuracy: {xgb_acc:.4f} (Training Time: {xgb_time:.2f}s)")
print(f"🎯 MLP (Deep Learning) Accuracy: {mlp_acc:.4f} (Training Time: {mlp_time:.2f}s)")

# 📌 9. Οπτικοποίηση Αποτελεσμάτων
models = ['Random Forest', 'XGBoost', 'MLP (Deep Learning)']
accuracies = [rf_acc, xgb_acc, mlp_acc]
times = [rf_time, xgb_time, mlp_time]

plt.figure(figsize=(12, 5))

# 🔹 Accuracy Plot
plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.8, 1)  # Ελαφρώς πάνω από το 80% για καλύτερη απεικόνιση

# 🔹 Training Time Plot
plt.subplot(1, 2, 2)
plt.bar(models, times, color=['blue', 'orange', 'green'])
plt.xlabel("Model")
plt.ylabel("Training Time (sec)")
plt.title("Model Training Time Comparison")

plt.tight_layout()
plt.show()
