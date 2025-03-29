from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import pandas as pd

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).values
label = (dataset_df['slice Type'] - 1).values

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

X_train, X_temp, y_train, y_temp = train_test_split(features, label, test_size=0.2, stratify=label, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

linearSVC = SVC(kernel='linear', probability=True, random_state=42)
linearSVC.fit(X_train,y_train)

train_predictions = linearSVC.predict(X_train)
test_predictions = linearSVC.predict(X_test)
val_predictions = linearSVC.predict(X_val)

test_accuracy = accuracy_score(y_test, test_predictions) * 100
train_accuracy = accuracy_score(y_train, train_predictions) * 100
val_accuracy = accuracy_score(y_val, val_predictions) * 100

joblib.dump(linearSVC, "LinearSVC_model/linearsvc_model.pkl")
joblib.dump(scaler, "LinearSVC_model/scaler.pkl")

print("\n================================\n")
print(classification_report(y_test, test_predictions, digits=2))
print("================================\n")
print(f"-> Train Accuracy: {train_accuracy:.2f}%\n")
print(f"-> Val Accuracy: {val_accuracy:.2f}%\n")
print(f"-> Test Accuracy: {test_accuracy:.2f}%\n")

cm = ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, cmap="Blues")
plt.title("LinearSVC - Confusion Matrix")
plt.grid(False)
plt.savefig("LinearSVC_CM/CM.png")

print("================================\n")
print("Plots saved:\n -> LinearSVC/LinearSVC_CM\n")
print("Scaler saved:\n -> LinearSVC/LinearSVC_model\n")
print("Model saved:\n -> LinearSVC/LinearSVC_model\n")
print("================================\n")
