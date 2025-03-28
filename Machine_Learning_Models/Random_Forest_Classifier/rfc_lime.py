# Import the LimeTabularExplainer module
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from dataset import features,X_train,X_test

with open('RandomForestClassifierModel.pkl', 'rb') as f:
    RandForestClass = pickle.load(f)
# Get the class names
class_names = ['1','2','3']
X_train = pd.DataFrame(X_train, columns = features.columns)
X_test = pd.DataFrame(X_test,columns=features.columns)

# Get the feature names
feature_names = list(X_train.columns)

# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names =     
                                 feature_names,
                                 class_names = class_names, 
                                 mode = 'classification')
def wrapped_fn(X):
    return RandForestClass.predict_proba(X)  # Ensure your model has `predict_proba`

# Number of instances to explain
num_instances = 20  # Change this as needed

for i in range(num_instances):    
    exp = explainer.explain_instance(X_test.values[i], RandForestClass.predict_proba, num_features=16)
    exp.save_to_file(f"lime/lime_exp{i}.html")
# Generate explanation for the second instance in X_test


# Show the explanation in a Jupyter Notebook
#fig = exp.as_pyplot_figure()
#fig.savefig("lime/rfc_lime.png")  # Save as PNG file
