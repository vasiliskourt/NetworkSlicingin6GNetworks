from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,  classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import joblib
import pandas as pd

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type'])
label = (dataset_df['slice Type'] - 1)

# Initialize scale and data normalization
scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(features, label, test_size=0.2, stratify=label, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Initialize model
randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)

# Training model
randomForestModel.fit(X_train,y_train)

# Predictions of model
train_predictions = randomForestModel.predict(X_train)
test_predictions = randomForestModel.predict(X_test)
val_predictions = randomForestModel.predict(X_val)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, test_predictions) * 100
train_accuracy = accuracy_score(y_train, train_predictions) * 100
val_accuracy = accuracy_score(y_val, val_predictions) * 100

# Save model and scaler 
joblib.dump(randomForestModel, "RandomForestClassifier_model/random_forest_model.pkl")
joblib.dump(scaler, "RandomForestClassifier_model/scaler.pkl")

# Print Results
print("\n================================\n")
print(classification_report(y_test, test_predictions, digits=2))
print("================================\n")
print(f"-> Train Accuracy: {train_accuracy:.2f}%\n")
print(f"-> Val Accuracy: {val_accuracy:.2f}%\n")
print(f"-> Test Accuracy: {test_accuracy:.2f}%\n")

# Generate Confusion Matrix
cm = ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.grid(False)
plt.savefig("RandomForestClassifier_CM/CM.png")

print("================================\n")
print("Plots saved:\n -> Random_Forest_Classifier/RandomForestClassifier_CM\n")
print("Scaler saved:\n -> Random_Forest_Classifier/RandomForestClassifier_model\n")
print("Model saved:\n -> Random_Forest_Classifier/RandomForestClassifier_model\n")
print("================================\n")
