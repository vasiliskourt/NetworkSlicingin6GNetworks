import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_score,recall_score, f1_score, accuracy_score
from sklearn.model_selection import  train_test_split
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import X_train,X_test,y_train,y_test

RandForestClass = LinearSVC()

RandForestClass.fit(X_train, y_train)


prediction = RandForestClass.predict(X_test)

accuracy = accuracy_score(y_test, prediction) * 100
print("Accuracy:", accuracy ,"%")
print("Precision:", precision_score(y_test,
                                    prediction,
                                    average="weighted"))

print('Recall:', recall_score(y_test,
                              prediction,
                              average="weighted"))
print(classification_report(prediction, y_test))


with open('RandomForestClassifierModel.pkl','wb') as f:
    pickle.dump(RandForestClass,f)
    print(f"Model saved as:'RandomForestClassifierModel.pkl'")

print("\n================================\n")
print(classification_report(y_test, prediction, digits=2))
print("================================\n")

#feature_importances = RandForestClass.feature_importances_
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
plt.savefig("linearSVC_CM/CM.png")


