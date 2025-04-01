from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    if not os.path.exists("RandomForestClassifier_model/random_forest_model.pkl"):
        raise FileNotFoundError("\n-> Model do not exist. Train Random Forest model First!\n")
    if not os.path.exists("RandomForestClassifier_model/scaler.pkl"):
        raise FileNotFoundError("\n-> Scaler do not exist. Train Random Forest model First!\n")

    model = joblib.load("RandomForestClassifier_model/random_forest_model.pkl")
    scaler = joblib.load("RandomForestClassifier_model/scaler.pkl")

    dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
    features = dataset_df.drop(columns=['slice Type'])
    label = (dataset_df['slice Type'] - 1).values

    features_scaled = scaler.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    print("-> Running SHAP...\n")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled_df)

    label_name = ['Slice_Type_1', 'Slice_Type_2', 'Slice_Type_3']

    for i, name in enumerate(label_name):
        plt.figure(figsize=(14, 7))
        shap.summary_plot(shap_values[:,:,i], features=features_scaled_df, show=False)
        plt.title(f"SHAP Summary Plot - {name}")
        plt.tight_layout()
        plt.savefig(f"SHAP/Summary_plot/shap_summary_plot_{name}.png", bbox_inches='tight')
        plt.close()

    shap_values_avg = np.mean(np.abs(shap_values), axis=2)
    
    plt.figure(figsize=(14, 7))
    shap.summary_plot(shap_values_avg, features=features_scaled_df, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("SHAP/Summary_plot/shap_summary_plot_avg.png", bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    for i, name in enumerate(label_name):
        plt.figure(figsize=(14, 7))
        shap.summary_plot(shap_values[:,:,i], features=features_scaled_df, plot_type="bar", show=False)
        plt.title(f"SHAP Bar Plot - {name}")
        plt.savefig(f"SHAP/Bar_plot/shap_bar_plot_{name}.png", bbox_inches='tight')
        plt.tight_layout()
        plt.close()

    plt.figure(figsize=(14, 7))
    shap.summary_plot(shap_values_avg, features=features_scaled_df, plot_type="bar", show=False)
    plt.title("SHAP Bar Plot")
    plt.savefig("SHAP/Bar_plot/shap_bar_plot_avg.png", bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    print("================================\n")
    print("Plots saved:\n -> Random_Forest_Classifier/SHAP\n")
    print("================================\n")

    shap_importance = np.mean(shap_values_avg, axis=0)
    features_importance_series = pd.Series(shap_importance, index=features.columns)
    feature_names_sorted = features_importance_series.sort_values(ascending=True).index

    print(features_importance_series.sort_values(ascending=False))
    print("================================\n")

    accuracies = []
    features_to_drop = []
    counter = 0
    print("-> Training model...\n")

    for feature_drop in feature_names_sorted:
        counter += 1

        if counter == 16:
            break

        features_to_drop.append(feature_drop)
        
        for column in features_to_drop:
            if column in features_scaled_df.columns:
                features_scaled_df.drop(columns=column, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(features_scaled_df, label, test_size=0.3, stratify=label, random_state=42)
        
        randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)

        randomForestModel.fit(X_train,y_train)

        test_predictions = randomForestModel.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_predictions) * 100

        accuracies.append(test_accuracy)

        print(f"-> Train with {len(features_scaled_df.columns)} features, accuracy: {test_accuracy:.2f}%\n")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accuracies)), accuracies, label='Test Accuracy')
    plt.title("Accuracy vs Number of Features")
    plt.xlabel("Number Features Used")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("RandomForest_drop_feature_plot/accuracy_features.png")

    print("================================\n")
    print("Plots saved:\n -> Random_Forest_Classifier/RandomForest_drop_feature_plot\n")
    print("================================\n")

except FileNotFoundError as e:
    print(e)

