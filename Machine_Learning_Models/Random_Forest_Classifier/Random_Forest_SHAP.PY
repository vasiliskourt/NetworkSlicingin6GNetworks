import shap
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


if not os.path.exists("RandomForestClassifier_model/random_forest_model.pkl"):
    raise FileNotFoundError("Model do not exist. Train Random Forest model First\n")
if not os.path.exists("RandomForestClassifier_model/scaler.pkl"):
    raise FileNotFoundError("Scaler do not exist. Train Random Forest model First\n")

model = joblib.load("RandomForestClassifier_model/random_forest_model.pkl")
scaler = joblib.load("RandomForestClassifier_model/scaler.pkl")


dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
features = dataset_df.drop(columns=['slice Type'])
label = (dataset_df['slice Type'] - 1).values

features_scaled = scaler.transform(features)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_scaled)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, features=features_scaled, feature_names=features.columns, show=False)
plt.title("SHAP Summary Plot - Random Forest")
plt.savefig("SHAP/shap_summary_plot.png")

print("================================\n")
print("Plots saved:\n -> Random_Forest_Classifier/SHAP\n")
print("================================\n")

