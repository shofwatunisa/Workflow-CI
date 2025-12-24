import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Argument Parser (WAJIB SAMA DENGAN CI)
# =========================
parser = argparse.ArgumentParser(description="Text Emotion Classification")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to dataset CSV"
)
parser.add_argument(
    "--target_column",
    type=str,
    default="label_encoded",
    help="Target column name"
)

args = parser.parse_args()

# =========================
# Load Dataset
# =========================
df = pd.read_csv(args.dataset_path)

X = df["clean_text"]
y = df[args.target_column]

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# Model Training
# =========================
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

# =========================
# MLflow Logging (INI YANG DULU HILANG)
# =========================
mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=X_test[:5],
)

print("✅ Training & MLflow logging SUCCESS")
