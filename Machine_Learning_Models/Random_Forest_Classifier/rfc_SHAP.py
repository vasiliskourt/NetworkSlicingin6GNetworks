import pickle
import shap
import matplotlib.pyplot as plt
from dataset import features, X_test

with open('RandomForestClassifierModel.pkl', 'rb') as f:
    RandForestClass = pickle.load(f)

# Create the SHAP Explainer
explainer = shap.TreeExplainer(RandForestClass)

shap_values = explainer.shap_values(features)
shap.summary_plot(shap_values, features, plot_type="bar", class_names=['Value:1', 'Value:2', 'Value:3'],show=False)
plt.savefig("SHAP/SHAP_Bar.png")

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test,show=False)
plt.savefig("SHAP/SHAP_beeswarm.png")

print("Figures saved successfully at SHAP folder.")
