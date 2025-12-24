# modelling.py
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

# -------------------------
# Argument Parser
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path ke CSV dataset")
parser.add_argument("--target_column", type=str, default="label", help="Nama kolom target")
parser.add_argument("--random_state", type=int, default=42, help="Random state untuk reproducibility")
args = parser.parse_args()

dataset_path = args.dataset_path
target_column = args.target_column
random_state = args.random_state

# -------------------------
# Load Dataset
# -------------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di path: {dataset_path}")

df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

if target_column not in df.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset")

X = df.drop(target_column, axis=1)
y = df[target_column]

# -------------------------
# Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

# -------------------------
# MLflow Experiment
# -------------------------
mlflow.set_experiment("TextEmotion_CI")

with mlflow.start_run():
    # -------------------------
    # Train Model
    # -------------------------
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    
    # -------------------------
    # Evaluate
    # -------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:", report)

    # -------------------------
    # Log Metrics & Model
    # -------------------------
    mlflow.log_metric("accuracy", acc)
    
    # Log per-class f1-score
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            mlflow.log_metric(f"f1_{label}", metrics["f1-score"])

    # Log Model
    mlflow.sklearn.log_model(clf, artifact_path="model")
    
    # Save test set as artifact
    test_df = X_test.copy()
    test_df[target_column] = y_test
    test_csv_path = "test_dataset.csv"
    test_df.to_csv(test_csv_path, index=False)
    mlflow.log_artifact(test_csv_path)
    
print("MLflow run completed successfully!")
