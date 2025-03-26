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
            "MLP K-Fold Cross-Validation",
            "CNN K-Fold Cross-Validation",
            "MLP Train",
            "CNN Train",
            "Exit"
        ]
    ).ask()

    if select == "MLP K-Fold Cross-Validation":
        executeScript("MLP_K_Fold.py", "Deep_Neural_Networks/MLP")

    elif select == "CNN K-Fold Cross-Validation":
        executeScript("CNN_K_Fold.py", "Deep_Neural_Networks/CNN")

    elif select == "MLP Train":
        executeScript("MLP.py", "Deep_Neural_Networks/MLP")

    elif select == "CNN Train":
        executeScript("CNN.py", "Deep_Neural_Networks/CNN")

    elif select == "Exit":
        break
