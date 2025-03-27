import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,  classification_report
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type']).values
label = (dataset_df['slice Type'] - 1).values

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(features)

X_train, X_temp, y_train, y_temp = train_test_split(features, label, test_size=0.2, stratify=label, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_dmatrix = xgb.DMatrix(X_train, y_train)
val_dmatrix = xgb.DMatrix(X_val, y_val)
test_dmatrix = xgb.DMatrix(X_test, y_test)

parameters = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'seed': 42
}

evals_res = {}

model = xgb.train(
    parameters,
    train_dmatrix,
    num_boost_round=100,
    evals=[(train_dmatrix,'train'),(val_dmatrix,'validation')],
    early_stopping_rounds=10,
    evals_result= evals_res,
    verbose_eval = True
)

plt.figure(figsize=(10, 4))
plt.plot(evals_res['train']['mlogloss'], label="Train Loss")
plt.plot(evals_res['validation']['mlogloss'], label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("XGBoost_train_plots/train_validation_accuracy.png")

predictions = np.argmax(model.predict(test_dmatrix), axis=1)
test_accuracy = accuracy_score(y_test, predictions) * 100

model.save_model("XGBoost_model/xgboost_model.json")

print(f"\n-> Test Accuracy: {test_accuracy:.2f}%")
print("================================\n")

print("Plots saved:\n -> XGBoost_Classifier/XGBoost_train_plots\n")
print("Model saved:\n -> XGBoost_Classifier/XGBoost_model\n")
print("================================\n")