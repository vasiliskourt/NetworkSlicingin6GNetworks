import questionary
import subprocess
import sys

def executeScript(script, directory):

    process = subprocess.Popen([sys.executable, "-u", script], cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")

    process.wait()

while True:
    select = questionary.select(
        "Select a choice",
        choices=[
            "Dataset Description",
            "MLP K-Fold Cross-Validation",
            "CNN K-Fold Cross-Validation",
            "Random Forest K-Fold Cross-Validation",
            "XGBoost K-Fold Cross-Validation",
            "MLP Train",
            "CNN Train",
            "Random Forest Train",
            "XGBoost Train",
            "Random Forest Classifier SHAP",
            "Random Forest Classifier Lime",
            "Exit"
        ]
    ).ask()

    if select == "Dataset Description":
        executeScript("dataset_description.py", "Dataset")

    elif select == "MLP K-Fold Cross-Validation":
        executeScript("MLP_K_Fold.py", "Deep_Neural_Networks/MLP")

    elif select == "CNN K-Fold Cross-Validation":
        executeScript("CNN_K_Fold.py", "Deep_Neural_Networks/CNN")

    elif select == "MLP Train":
        executeScript("MLP.py", "Deep_Neural_Networks/MLP")

    elif select == "CNN Train":
        executeScript("CNN.py", "Deep_Neural_Networks/CNN")

    elif select == "Random Forest K-Fold Cross-Validation":
        executeScript("Random_Forest_Classifier_K_Fold.py","Machine_Learning_Models/Random_Forest_Classifier")

    elif select == "XGBoost K-Fold Cross-Validation":
        executeScript("XGBoost_Classifier_K_Fold.py","Machine_Learning_Models/XGBoost_Classifier")
    
    elif select == "Random Forest Train":
        executeScript("Random_Forest_Classifier.py","Machine_Learning_Models/Random_Forest_Classifier")

    elif select == "XGBoost Train":
        executeScript("XGBoost_Classifier.py","Machine_Learning_Models/XGBoost_Classifier")

    elif select == "Random Forest Classifier SHAP":
        executeScript("rfc_SHAP.py","Machine_Learning_Models/Random_Forest_Classifier")

    elif select == "Random Forest Classifier Lime":
        executeScript("rfc_lime.py","Machine_Learning_Models/Random_Forest_Classifier")

    elif select == "Exit":
        break
