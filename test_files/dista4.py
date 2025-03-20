import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 📌 1. Φόρτωση του dataset
df = pd.read_csv("train_dataset.csv")

# 📌 2. Διαχωρισμός χαρακτηριστικών και labels
X = df.drop(columns=['slice Type']).values
y = df['slice Type'].values
y = y - 1  # 🔹 Labels από 0, 1, 2

# 📌 3. Κανονικοποίηση δεδομένων
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 4. Διαχωρισμός σε train και test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 📌 5. Εκπαίδευση του XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# 📌 6. Υπολογισμός SHAP Values
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# 📌 7. Οπτικοποίηση σημαντικότητας χαρακτηριστικών
shap.summary_plot(shap_values, X_test, feature_names=df.drop(columns=['slice Type']).columns)

# 📌 1. Βρίσκουμε τη σημασία των χαρακτηριστικών
importance = np.abs(shap_values.values).mean(axis=0).flatten()
feature_names = df.drop(columns=['slice Type']).columns.tolist()  # Παίρνουμε τα ονόματα των χαρακτηριστικών
importance = np.abs(shap_values.values).mean(axis=0)  # Υπολογισμός SHAP values

# 🔹 Ελέγχουμε αν οι διαστάσεις ταιριάζουν
if len(feature_names) == len(importance):
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print("🔹 Σημαντικότητα Χαρακτηριστικών σύμφωνα με SHAP:")
    print(feature_importance)
else:
    print(f"❌ Error: Feature names ({len(feature_names)}) and SHAP importance values ({len(importance)}) do not match!")

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("🔹 Σημαντικότητα χαρακτηριστικών σύμφωνα με SHAP:")
print(feature_importance)

# 📌 2. Αφαιρούμε σταδιακά τα λιγότερο σημαντικά χαρακτηριστικά
accuracies = []
num_features_list = []

for num_features in range(16, 3, -2):  # Ξεκινάμε με όλα και μειώνουμε ανά 2 χαρακτηριστικά
    selected_features = feature_importance['Feature'].iloc[:num_features].values
    X_reduced = df[selected_features].values
    X_scaled = scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    accuracies.append(acc)
    num_features_list.append(num_features)

    print(f"📉 Features used: {num_features} | Accuracy: {acc:.4f}")

# 📌 3. Οπτικοποίηση της μείωσης της ακρίβειας
plt.figure(figsize=(10, 5))
plt.plot(num_features_list, accuracies, marker='o', linestyle='-')
plt.xlabel("Number of Features Used")
plt.ylabel("Accuracy")
plt.title("Impact of Feature Reduction on Accuracy")
plt.grid()
plt.show()

