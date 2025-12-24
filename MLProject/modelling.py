import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# =========================
# Load dataset
# =========================
df = pd.read_csv(args.dataset_path)

X = df["clean_text"]      
y = df["label_encoded"]     

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state, stratify=y
)

# =========================
# ML Pipeline
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=500))
])

# =========================
# MLflow Tracking
# =========================
with mlflow.start_run(run_name="Text Emotion Classification"):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    # ---- log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ---- save report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # ---- log model (INI YANG KEMARIN HILANG)
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model"
    )

print("TRAINING SELESAI & MODEL TERSIMPAN DI MLflow")
