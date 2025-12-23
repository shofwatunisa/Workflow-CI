import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dataset_preprocessing/text_emotion_clean.csv")
parser.add_argument("--max_features", type=int, default=5000)
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# MLflow tracking
mlflow.set_experiment("Text Emotion Classification")

# LOAD DATA
df = pd.read_csv(args.data_path)
X = df["clean_text"].fillna("")
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:

    # Log params
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", args.max_iter)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train_vec, y_train)

    # Predict & metrics
    y_pred = model.predict(X_test_vec)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))

    # Log model with signature
    signature = infer_signature(X_train_vec, model.predict(X_train_vec))
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=X_train_vec[:5])

    # Log vectorizer
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    mlflow.log_artifact("tfidf_vectorizer.pkl")

    # Log sample data
    sample_df = df.sample(100)
    sample_df.to_csv("sample_data.csv", index=False)
    mlflow.log_artifact("sample_data.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    print("Training SUCCESS - BASE MODEL")