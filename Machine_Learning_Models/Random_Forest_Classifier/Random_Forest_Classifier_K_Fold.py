from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type'])
label = (dataset_df['slice Type'] - 1)

# Initialize scale and data normalization 
scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

# Number of Folds
k_folds_n = 5
fold_n = 0

# Initialize Stratified K-Fold
k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

val_fold_accuracies = []
train_fold_accuracies = []
train_time_l = []

classific_report = []

for train_index, val_index in k_folds.split(features, label):

    fold_n += 1
    # Array of every fold
    X_train = features[train_index]
    X_val = features[val_index]
    y_train = label[train_index]
    y_val = label[val_index]

    #Load the model and parameters
    randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
    #Calculate training time
    train_time_start = time.time()
    #Training of model
    randomForestModel.fit(X_train,y_train)  

    train_time_end = time.time()
    training_time = train_time_end - train_time_start

    #Prediction of validation and training data
    val_predictions = randomForestModel.predict(X_val)
    train_predictions = randomForestModel.predict(X_train)

    #Calculate  validation and train accuracy
    val_accuracy = accuracy_score(y_val, val_predictions) * 100
    train_accuracy = accuracy_score(y_train, train_predictions) * 100

    train_time_l.append(training_time)
    val_fold_accuracies.append(val_accuracy)
    train_fold_accuracies.append(train_accuracy)

    print(f"-> Fold {fold_n} Validation Accuracy: {val_accuracy:.2f}%, Train Accuracy: {train_accuracy:.2f}%, Training Time: {training_time:.3f} seconds\n")

    classific_report.append(classification_report(y_val,val_predictions, digits=2))

    cm = ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, cmap="Blues")
    plt.title("Random Forest - Confusion Matrix")
    plt.grid(False)
    plt.savefig(f"RandomForestClassifier_K_Fold_plots/CM_{fold_n}.png")

#Calculate mean of folds' accuracy for validation and training
avg_val_accuracy = np.mean(val_fold_accuracies)
avg_train_accuracy = np.mean(train_fold_accuracies)

# Generate Train Time
plt.figure(figsize=(10, 4))
plt.plot(range(1, k_folds_n + 1), train_time_l, label="Time")
plt.title("(Random Forest) Time to train")
plt.xlabel("Fold")
plt.ylabel("Training Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(f"RandomForestClassifier_K_Fold_plots/training_time.png")

# Generate Folds' Train/Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds_n + 1), val_fold_accuracies, label="Val Accuracy")
plt.plot(range(1, k_folds_n + 1), train_fold_accuracies, label="Train Accuracy")
plt.title("(Random Forest) Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(range(1, k_folds_n + 1))
plt.legend()
plt.grid(True)
plt.savefig(f"RandomForestClassifier_K_Fold_plots/k_folds_accuracy.png")

# Save k fold result to report
with open("RandomForest_report/randomforest_report.txt", "w") as file:
    file.write("--------- Random Forest Report---------\n")
    file.write("\n-> Validation Accuracy:\n")
    for i, acc in enumerate(val_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\nAverage Validation Accuracy: {np.mean(val_fold_accuracies):.2f}%\n")
    file.write("\n-> Train Accuracy:\n")
    for i, acc in enumerate(train_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\nAverage Train Validation Accuracy: {np.mean(train_fold_accuracies):.2f}%\n")
    file.write(f"\n-> Train Time:\n")
    for i, times in enumerate(train_time_l):
        file.write(f"Fold {i+1}: {times:.3f} seconds\n")
    file.write(f"\nAverage Train Time: {np.mean(train_time_l):.3f} seconds\n")
    file.write(f"\n-> Classification Report\n")
    for i, report in enumerate(classific_report):
        file.write(f"Fold {i+1}:\n {report}\n")


print("================================\n")
print(f"-> Average Validation Accuracy: {avg_val_accuracy:.2f}%\n")
print(f"-> Average Train Accuracy: {avg_train_accuracy:.2f}%\n")
print("================================\n")
print("Report saved:\n -> Random_Forest_Classifier/RandomForest_report\n")
print("Plots saved:\n -> Random_Forest_Classifier/RandomForestClassifier_K_Fold_plots\n")
print("================================\n")

