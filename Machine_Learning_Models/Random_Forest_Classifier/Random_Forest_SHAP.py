from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    if not os.path.exists("RandomForestClassifier_model/random_forest_model.pkl"):
        raise FileNotFoundError("\n-> Model do not exist. Train Random Forest model First!\n")
    if not os.path.exists("RandomForestClassifier_model/scaler.pkl"):
        raise FileNotFoundError("\n-> Scaler do not exist. Train Random Forest model First!\n")

    # Load model and scaler
    model = joblib.load("RandomForestClassifier_model/random_forest_model.pkl")
    scaler = joblib.load("RandomForestClassifier_model/scaler.pkl")

    dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")
    features = dataset_df.drop(columns=['slice Type'])
    label = (dataset_df['slice Type'] - 1).values

    # Normalize features
    features_scaled = scaler.transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    # Generate feature importances
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=features.columns)

    # Print feature importances
    print("-> Feature importances:\n")
    print(forest_importances.sort_values(ascending=False))
    print("================================\n")

    # Generate plot of feature importances
    plt.figure(figsize=(14, 7))
    forest_importances.sort_values().plot(kind='barh')
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("Feature_Importances/feature_importances_plot.png")
    plt.close()

    # SHAP Calculation
    print("-> Running SHAP...\n")

    # Build Tree explainer
    explainer = shap.TreeExplainer(model)

    # Estimate SHAP values
    shap_values = explainer.shap_values(features_scaled_df)

    label_name = ['Slice_Type_1', 'Slice_Type_2', 'Slice_Type_3']

    # Generate SHAP plot
    for i, name in enumerate(label_name):
        plt.figure(figsize=(14, 7))
        shap.summary_plot(shap_values[:,:,i], features=features_scaled_df, show=False)
        plt.title(f"SHAP Summary Plot - {name}")
        plt.tight_layout()
        plt.savefig(f"SHAP/Summary_plot/shap_summary_plot_{name}.png", bbox_inches='tight')
        plt.close()

    # Calculate avg SHAP values for each slice
    shap_values_avg = np.mean(np.abs(shap_values), axis=2)
    
    # Plot SHAP values across all slices
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

    # Calculate shap importance and print sorted
    shap_importance = np.mean(shap_values_avg, axis=0)
    features_importance_series = pd.Series(shap_importance, index=features.columns)
    feature_names_sorted = features_importance_series.sort_values(ascending=True).index

    print("-> SHAP Feature Importances:\n")
    print(features_importance_series.sort_values(ascending=False))
    print("================================\n")

    accuracies = []
    features_to_drop = []
    counter = 0
    print("-> Training model...\n")

    time_l = []

    # Train model dropping a feature
    for feature_drop in feature_names_sorted:
        counter += 1

        if counter == 16:
            break

        # Delete features
        features_to_drop.append(feature_drop)
        
        for column in features_to_drop:
            if column in features_scaled_df.columns:
                features_scaled_df.drop(columns=column, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(features_scaled_df, label, test_size=0.3, stratify=label, random_state=42)
        
        randomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)

        train_time_start = time.time()
        randomForestModel.fit(X_train,y_train)
        train_time_end = time.time()

        time_train = train_time_end-train_time_start

        time_l.append(time_train)

        test_predictions = randomForestModel.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_predictions) * 100

        accuracies.append(test_accuracy)

        print(f"-> Train with {len(features_scaled_df.columns)} features, accuracy: {test_accuracy:.2f}%, Training time: {time_train:.3f} seconds\n")

    # Generate Train Time
    plt.figure(figsize=(10, 4))
    plt.plot(list(range(15, 0, -1)), time_l, label="Time") 
    plt.title("(Random Forest) Time to train")
    plt.xlabel("Number of Features")
    plt.ylabel("Training Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"RandomForest_drop_feature_plot/training_time.png")

    # Plot Accuracy of every train
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(len(accuracies), 0, -1)), accuracies, label='Test Accuracy')
    plt.gca().invert_xaxis()
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

