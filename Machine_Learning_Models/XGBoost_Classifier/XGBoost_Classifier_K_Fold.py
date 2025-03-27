import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type']).values
label = (dataset_df['slice Type'] - 1).values

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

k_folds_n = 5

k_folds = StratifiedKFold(n_splits=k_folds_n, shuffle=True, random_state=42)
fold_n = 0

fold_accuracies = []

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

    model = xgb.train(parameters, train_dmatrix,
                      num_boost_round=100,
                      verbose_eval=False)
    
    predictions = np.argmax(model.predict(val_dmatrix), axis=1)
    accuracy = accuracy_score(y_val, predictions)

    fold_accuracies.append(accuracy)

    print(f"-> Fold {fold_n} Average Validation Accuracy: {accuracy * 100:.2f}%\n")

avg_accuracy = np.mean(fold_accuracies)
