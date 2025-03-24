import questionary
import subprocess
import sys


while True:
    select = questionary.select(
        "Select:",
        choices=[
            "MLP K-Fold Cross-Validation",
            "MLP Train",
            "Exit"
        ]
    ).ask()

    if select == "MLP K-Fold Cross-Validation":
        print("Executing MLP K-Fold Cross-Validation...\n")
        result = subprocess.run([sys.executable, "MLP_K_Fold.py"], cwd="Deep_Neural_Networks/MLP",
        capture_output=True,text=True)
        print(result.stdout)

    elif select == "MLP Train":
        print("Executing MLP model training...\n")
        result = subprocess.run([sys.executable, "MLP.py"], cwd="Deep_Neural_Networks/MLP",
        capture_output=True,text=True)
        print(result.stdout)

    elif select == "Exit":
        break
