import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default="text_emotion_preprocessing/text_emotion_clean.csv"
)
args = parser.parse_args()

# =========================
# Load Dataset
# =========================
df = pd.read_csv(args.dataset_path)

X = df["clean_text"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Feature Engineering
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# Train Model (MLflow)
# =========================
with mlflow.start_run(run_name="Text Emotion Classification"):

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", acc)

    # Classification report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    plt.close()

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="TextEmotionModel"
    )

print("Training selesai & model tercatat di MLflow")
