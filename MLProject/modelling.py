import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser(description="Text Emotion Classification Training with MLflow")

parser.add_argument(
    "--dataset_path",
    type=str,
    default="MLProject/text_emotion_preprocessing/text_emotion_clean.csv",
    help="Path to cleaned dataset"
)

parser.add_argument(
    "--text_column",
    type=str,
    default="clean_text",
    help="Text column name"
)

parser.add_argument(
    "--target_column",
    type=str,
    default="label_encoded",
    help="Target column name"
)

parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="Random state"
)

args = parser.parse_args()


# =========================
# Load Dataset
# =========================
if not os.path.exists(args.dataset_path):
    raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

df = pd.read_csv(args.dataset_path)

for col in [args.text_column, args.target_column]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")

df = df.dropna(subset=[args.text_column, args.target_column])

X = df[args.text_column]
y = df[args.target_column]


# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=args.random_state,
    stratify=y
)


# =========================
# MLflow Experiment
# =========================
mlflow.set_experiment("Text Emotion Classification")

with mlflow.start_run():

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=500, random_state=args.random_state)
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print(report)

    # =========================
    # MLflow Logging (WAJIB)
    # =========================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # Simpan report
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

print("TRAINING SELESAI. MODEL TERCATAT DI MLflow.")
