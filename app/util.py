import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_model():
    command = "python model.py"
    subprocess.run(command, shell=True)

def get_predictions():
    # Read predictions from a CSV file
    df = pd.read_csv("output/predictions.csv")
    return df

def evaluate_model(df):
    y_pred = df["Prediction"]
    ground_truth = np.array(open(f"ground_truth.txt").read().strip().split("\n"))

    accuracy = accuracy_score(ground_truth, y_pred)
    precision = precision_score(ground_truth, y_pred, average="weighted")
    recall = recall_score(ground_truth, y_pred, average="weighted")
    f1 = f1_score(ground_truth, y_pred, average="weighted")

    return accuracy, precision, recall, f1
    
def save_uploaded(uploaded_file, filename):
    with open(f"{filename}", "wb") as f:
        f.write(uploaded_file.getvalue())