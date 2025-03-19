import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import optuna

# 📌 1. Φόρτωση του dataset
df = pd.read_csv("train_dataset.csv")

# 📌 2. Διαχωρισμός χαρακτηριστικών και labels
X = df.drop(columns=['slice Type']).values
y = df['slice Type'].values  # Κατηγορίες (1, 2, 3)

# 📌 3. Κανονικοποίηση των χαρακτηριστικών
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 📌 4. Μετατροπή σε PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long) - 1  # Labels ξεκινούν από 0

# 📌 5. Διαχωρισμός σε train και test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y)

# 📌 6. Δημιουργία PyTorch DataLoader (batch_size θα το βρούμε με Optuna)
def get_data_loaders(batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 📌 7. Ορισμός του MLP μοντέλου
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

# 📌 8. Συνάρτηση βελτιστοποίησης στο Optuna
def objective(trial):

    # 🔹 Επιλογή υπερπαραμέτρων από το Optuna (διορθωμένη)
    lr = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)  # Νέο (αντί για suggest_loguniform)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout',0.1, 0.5)  # Νέο (αντί για suggest_uniform)
    hidden_units = trial.suggest_categorical('hidden_units', [16, 32, 64, 128])

    # 🔹 Δημιουργία των DataLoaders
    train_loader, test_loader = get_data_loaders(batch_size)

    # 🔹 Δημιουργία του MLP μοντέλου
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=16, hidden_units=hidden_units, dropout=dropout, num_classes=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 🔹 Εκπαίδευση του μοντέλου (μόνο για 5 epochs για ταχύτητα)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    # 🔹 Υπολογισμός τελικής ακρίβειας στο test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100 * correct / total
    return -accuracy  # Θέλουμε να μεγιστοποιήσουμε την ακρίβεια (οπότε παίρνουμε το αρνητικό της)

# 📌 9. Εκτέλεση του Optuna για 20 διαφορετικές δοκιμές
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# 📌 10. Εκτύπωση του καλύτερου συνδυασμού υπερπαραμέτρων
print("Best hyperparameters:", study.best_params)

# 📌 Αποθήκευση του εκπαιδευμένου μοντέλου
torch.save(model.state_dict(), "mlp_best_model.pth")
print("✅ Model saved successfully as 'mlp_best_model.pth'")
