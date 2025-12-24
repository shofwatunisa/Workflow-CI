import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import argparse
import os

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Text Emotion Classification")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="MLProject/text_emotion_preprocessing/text_emotion_clean.csv",  # default dataset path
    help="Path to the cleaned dataset CSV file"
)
parser.add_argument(
    "--target_column",
    type=str,
    default="label",
    help="Name of the target column"
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="Random state for reproducibility"
)

args = parser.parse_args()
dataset_path = args.dataset_path
target_column = args.target_column
random_state = args.random_state

# -----------------------------
# Load Dataset
# -----------------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

df = pd.read_csv(dataset_path)

if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset columns.")

X = df.drop(target_column, axis=1)
y = df[target_column]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

# -----------------------------
# Feature Extraction
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train['text'])
X_test_tfidf = tfidf.transform(X_test['text'])

# -----------------------------
# Model Training
# -----------------------------
mlflow.set_experiment("TextEmotion_CI")

with mlflow.start_run():
    model = LogisticRegression(random_state=random_state, max_iter=500)
    model.fit(X_train_tfidf, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print("Classification Report:")
    print(report)

    # Log parameters and metrics
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")
