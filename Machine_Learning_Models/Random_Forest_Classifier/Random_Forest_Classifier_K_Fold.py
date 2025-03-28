from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,  classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type']).values
label = (dataset_df['slice Type'] - 1).values

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

k_folds_n = 5
fold_n = 0

k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

fold_accuracies = []

for train_index, val_index in k_folds.split(features, label):

    fold_n += 1

    X_train = features[train_index]
    X_val = features[val_index]
    y_train = label[train_index]
    y_val = label[val_index]

    randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
    randomForestModel.fit(X_train,y_train)  

    predictions = randomForestModel.predict(X_val)
    accuracy = accuracy_score(y_val, predictions) * 100

    fold_accuracies.append(accuracy)

    print(f"-> Fold {fold_n} Validation Accuracy: {accuracy:.2f}%\n")

    cm = ConfusionMatrixDisplay.from_predictions(y_val, predictions, cmap="Blues")
    plt.title("Random Forest - Confusion Matrix")
    plt.grid(False)
    plt.savefig(f"RandomForestClassifier_K_Fold_plots/CM_{fold_n}.png")

avg_accuracy = np.mean(fold_accuracies)

print("================================\n")
print(f"-> Average Validation Accuracy: {avg_accuracy:.2f}%\n")
print("================================\n")
print("Plots saved:\n -> Random_Forest_Classifier/RandomForestClassifier_K_Fold_plots\n")
print("================================\n")

