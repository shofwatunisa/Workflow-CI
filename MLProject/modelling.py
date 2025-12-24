import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description="Text Emotion Classification Training")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="MLProject/text_emotion_preprocessing/text_emotion_clean.csv"
)
parser.add_argument(
    "--text_column",
    type=str,
    default="clean_text"
)
parser.add_argument(
    "--target_column",
    type=str,
    default="label_encoded"
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42
)

args = parser.parse_args()

# ----------------------------
# Load dataset
# ----------------------------
if not os.path.exists(args.dataset_path):
    raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

df = pd.read_csv(args.dataset_path)

# ----------------------------
# Validate columns
# ----------------------------
for col in [args.text_column, args.target_column]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataset")

df = df.dropna(subset=[args.text_column, args.target_column])

X = df[args.text_column]
y = df[args.target_column]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=args.random_state,
    stratify=y
)

# ----------------------------
# Feature extraction
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


mlflow.set_experiment("Text Emotion Classification")

# log parameters
mlflow.log_param("model", "LogisticRegression")
mlflow.log_param("max_features", 5000)
mlflow.log_param("random_state", args.random_state)

# train model
model = LogisticRegression(max_iter=500, random_state=args.random_state)
model.fit(X_train_tfidf, y_train)

# evaluation
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

mlflow.log_metric("accuracy", acc)

# save artifacts
os.makedirs("artifacts", exist_ok=True)

model_path = "artifacts/text_emotion_model.pkl"
vectorizer_path = "artifacts/tfidf_vectorizer.pkl"
report_path = "artifacts/classification_report.txt"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

with open(report_path, "w") as f:
    f.write(classification_report(y_test, y_pred))

# log artifacts
mlflow.log_artifact(model_path)
mlflow.log_artifact(vectorizer_path)
mlflow.log_artifact(report_path)

print("Training SUCCESS")
print("Accuracy:", acc)
