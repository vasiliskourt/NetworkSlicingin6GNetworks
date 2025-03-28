import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pickle 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import features,label

k_folds_n = 5

k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)

fold_accuracies = []
fold_n = 0

for train_index, val_index in k_folds.split(features,label):
    fold_n += 1

    X_train = features[train_index]
    y_train = label[train_index]
    X_val = features[val_index]
    y_val = label[val_index]

    RandomForestClassKFold = RandomForestClassifier()
    RandomForestClassKFold.fit(X_train, y_train)

    y_pred = RandomForestClassKFold.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    # Confusion Matrix for the last fold
    cm = confusion_matrix(y_val, y_pred)
    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1','Class 2'], yticklabels=['Class 0', 'Class 1','Class 2'])
    plt.title(f'Confusion Matrix for Fold {fold_n}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"RandomForestClassifier_K_Fold_plots/CM_{fold_n}.png")
    print(f"-> Fold {fold_n} | Accuracy: {accuracy*100:.2f}%")

mean_acc = np.mean(fold_accuracies) * 100

print(f"Mean Accuracy Measured: {mean_acc:.2f}%")


with open('RandomForestClassifierKFoldModel.pkl','wb') as f:
    pickle.dump(RandomForestClassKFold,f)
    print(f"Model saved as:'RandomForestClassifierKFoldModel.pkl'")


