import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📌 1. Φόρτωση του dataset
df = pd.read_csv("train_dataset.csv")

# 📌 2. Διαχωρισμός χαρακτηριστικών και labels
X = df.drop(columns=['slice Type']).values
y = df['slice Type'].values  # Κατηγορίες (1, 2, 3)

# 🔹 Διόρθωση: Μετατροπή labels ώστε να ξεκινούν από 0 (0, 1, 2 αντί για 1, 2, 3)
y = y - 1

# 📌 3. Κανονικοποίηση των χαρακτηριστικών (απαραίτητο για XGBoost)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 4. Διαχωρισμός σε train και test sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 📌 5. Εκπαίδευση Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 📌 6. Εκπαίδευση XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# 📌 7. Αξιολόγηση Απόδοσης
rf_acc = accuracy_score(y_test, rf_preds)
xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"🎯 Random Forest Accuracy: {rf_acc:.4f}")
print(f"🎯 XGBoost Accuracy: {xgb_acc:.4f}")

print("\n🔹 Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))

print("\n🔹 XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))
