import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import features, label

k_folds_n = 5
k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

fold_accuracies = []
fold_n = 0

for train_index, val_index in k_folds.split(features, label):
    fold_n += 1

    # If features or label are pandas DataFrame or Series, use .values or .iloc
    X_train = features.iloc[train_index].values if isinstance(features, pd.DataFrame) else features[train_index]
    y_train = label.iloc[train_index].values if isinstance(label, pd.Series) else label[train_index]
    
    X_val = features.iloc[val_index].values if isinstance(features, pd.DataFrame) else features[val_index]
    y_val = label.iloc[val_index].values if isinstance(label, pd.Series) else label[val_index]

    linearSVCKFold = LinearSVC()
    linearSVCKFold.fit(X_train, y_train)

    y_pred = linearSVCKFold.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    
    # Confusion Matrix for the last fold
    cm = confusion_matrix(y_val, y_pred)
    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix for Fold {fold_n}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"linearSVC_K_Fold_plots/CM_{fold_n}.png")
    print(f"-> Fold {fold_n} | Accuracy: {accuracy*100:.2f}%")

mean_acc = np.mean(fold_accuracies) * 100
print(f"Mean Accuracy Measured: {mean_acc:.2f}%")

with open('linearSVCKFoldModel.pkl', 'wb') as f:
    pickle.dump(linearSVCKFold, f)
    print(f"Model saved as:'linearSVCKFoldModel.pkl'")