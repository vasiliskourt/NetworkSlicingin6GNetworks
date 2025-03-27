import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score
from sklearn.model_selection import  train_test_split
import pickle
from sklearn.metrics import confusion_matrix
import shap
import seaborn as sns
import matplotlib.pyplot as plt

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type'])
label = dataset_df['slice Type'] - 1

mMScaler = MinMaxScaler(feature_range=(0,1))
scaledFeatures = mMScaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(scaledFeatures, label, test_size=0.2, random_state=42, stratify=label)

RandForestClass = RandomForestClassifier()

RandForestClass.fit(X_train, y_train)

# Create the SHAP Explainer
explainer = shap.Explainer(RandForestClass)
treeExplainer = shap.TreeExplainer(RandForestClass)
shap_values = explainer.shap_values(features)

# Plot SHAP values for each class
#shap.summary_plot(shap_values, features, plot_type="bar", class_names=['Value:1', 'Value:2', 'Value:3'])
shap.summary_plot(shap_values, features)
prediction = RandForestClass.predict(X_test)

accuracy = accuracy_score(y_test, prediction) * 100
print("Accuracy:", accuracy ,"%")
print("Precision:", precision_score(y_test,
                                    prediction,
                                    average="weighted"))

print('Recall:', recall_score(y_test,
                              prediction,
                              average="weighted"))
with open('RandomForestClassifierModel.pkl','wb') as f:
    pickle.dump(RandForestClass,f)
    print(f"Model saved as:'RandomForestClassifierModel.pkl'")


feature_importances = RandForestClass.feature_importances_
'''
plt.barh(features.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.savefig("featureimportances.png")'
'''

cm = confusion_matrix(y_test, prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("RandomForestClassifier_CM/CM.png")


