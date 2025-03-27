import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type'])
label = dataset_df['slice Type'] - 1

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

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

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)

    print(f"-> Fold {fold_n} | Accuracy: {accuracy*100:.2f}%")

mean_acc = np.mean(fold_accuracies) * 100

print(f"Mean Accuracy Measured: {mean_acc:.2f}%")