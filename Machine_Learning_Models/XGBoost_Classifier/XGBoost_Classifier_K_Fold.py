import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type'])
label = (dataset_df['slice Type'] - 1)

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

k_folds_n = 5
fold_n = 0

k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

val_fold_accuracies = []
train_fold_accuracies = []
train_time_l = []
train_loss = []
val_loss = []
classific_report = []


for train_index, val_index in k_folds.split(features, label):
    fold_n += 1

    X_train = features[train_index]
    X_val = features[val_index]
    y_train = label[train_index]
    y_val = label[val_index]

    train_dmatrix = xgb.DMatrix(X_train,y_train)
    val_dmatrix = xgb.DMatrix(X_val,y_val)

    parameters = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    evals_res = {}

    train_time_start = time.time()

    model = xgb.train(parameters, train_dmatrix,
                      num_boost_round=100,
                      evals=[(train_dmatrix,'train'),(val_dmatrix,'validation')],
                      early_stopping_rounds=10,
                      evals_result= evals_res,
                      verbose_eval=False)
    
    train_time_end = time.time()
    training_time = train_time_end - train_time_start
    
    val_predictions = np.argmax(model.predict(val_dmatrix), axis=1)
    train_predictions = np.argmax(model.predict(train_dmatrix), axis=1)
    
    val_accuracy = accuracy_score(y_val, val_predictions) * 100
    train_accuracy = accuracy_score(y_train, train_predictions) * 100

    train_loss.append(np.mean(evals_res['train']['mlogloss']))
    val_loss.append(np.mean(evals_res['validation']['mlogloss']))
    val_fold_accuracies.append(val_accuracy)
    train_fold_accuracies.append(train_accuracy)
    train_time_l.append(training_time)

    plt.figure(figsize=(10, 4))
    plt.plot(evals_res['train']['mlogloss'], label="Train Loss")
    plt.plot(evals_res['validation']['mlogloss'], label="Validation Loss")
    plt.title(f"(XGBoost) Fold {fold_n} Loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"XGBoost_K_Fold_plots/train_validation_loss_fold_{fold_n}.png")

    print(f"-> Fold {fold_n} Validation Accuracy: {val_accuracy:.2f}%, Train Accuracy: {train_accuracy:.2f}%, Training Time: {training_time:.3f} seconds\n")

    classific_report.append(classification_report(y_val,val_predictions, digits=2))

    cm = ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, cmap="Blues")
    plt.title("Random Forest - Confusion Matrix")
    plt.grid(False)
    plt.savefig(f"XGBoost_K_Fold_plots/CM_{fold_n}.png")

avg_val_accuracy = np.mean(val_fold_accuracies)
avg_train_accuracy = np.mean(train_fold_accuracies)

plt.figure(figsize=(8, 5))
plt.plot(range(1, k_folds_n + 1), val_fold_accuracies, label="Val Accuracy")
plt.plot(range(1, k_folds_n + 1), train_fold_accuracies, label="Train Accuracy")
plt.title("(XGBoost) Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.xticks(range(1, k_folds_n + 1))
plt.legend()
plt.grid(True)
plt.savefig(f"XGBoost_K_Fold_plots/k_folds_accuracy.png")

plt.figure(figsize=(10, 4))
plt.plot(range(1, k_folds_n + 1), train_time_l, label="Time")
plt.title("(XGBoost) Time to train")
plt.xlabel("Fold")
plt.ylabel("Training Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(f"XGBoost_K_Fold_plots/training_time.png")

with open("XGBoost_report/xgboost_report.txt", "w") as file:
    file.write("---------XGBoost Report---------\n")
    file.write("\n-> Validation Accuracy:\n")
    for i, acc in enumerate(val_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\nAverage Validation Accuracy: {np.mean(val_fold_accuracies):.2f}%\n")
    file.write("\n-> Train Accuracy:\n")
    for i, acc in enumerate(train_fold_accuracies):
        file.write(f"Fold {i+1}: {acc:.2f}%\n")
    file.write(f"\nAverage Train Accuracy: {np.mean(train_fold_accuracies):.2f}%\n")
    file.write(f"\n-> Train Time:\n")
    for i, times in enumerate(train_time_l):
        file.write(f"Fold {i+1}: {times:.3f} seconds\n")
    file.write(f"\nAverage Train Time: {np.mean(train_time_l):.3f} seconds\n")
    file.write("\n-> Train Loss: (Avg per fold)\n")
    for i, loss in enumerate(train_loss):
        file.write(f"Fold {i+1}: {loss:.10f}\n")
    file.write(f"\nAverage Train Loss: {np.mean(train_loss):.10f}\n")
    file.write("\n-> Validation Loss (Avg per fold):\n")
    for i, loss in enumerate(val_loss):
        file.write(f"Fold {i+1}: {loss:.10f}\n")
    file.write(f"\nAverage Validation Loss: {np.mean(val_loss):.10f}\n")
    file.write(f"\n-> Classification Report\n")
    for i, report in enumerate(classific_report):
        file.write(f"Fold {i+1}:\n {report}\n")

print("================================\n")
print(f"-> Average Validation Accuracy: {avg_val_accuracy:.2f}%\n")
print(f"-> Average Train Accuracy: {avg_train_accuracy:.2f}%\n")
print(f"-> Average Validation Loss: {np.mean(val_loss):.10f}\n")
print(f"-> Average Train Loss: {np.mean(train_loss):.10f}\n")
print("================================\n")
print("Report saved:\n -> XGBoost_Classifier/XGBoost_report\n")
print("Plots saved:\n -> XGBoost_Classifier/XGBoost_K_Fold_plots\n")
print("================================\n")

