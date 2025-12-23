import os
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


MODE = os.getenv("MLFLOW_MODE", "local")
TOKEN = os.getenv("DMLFLOW_TRACKING_TOKEN")  

print("MLFLOW_TRACKING_URI:", mlflow.get_tracking_uri())
print("MLFLOW_MODE:", MODE)
print("Tracking URI:", mlflow.get_tracking_uri())

experiment_name = "Text Emotion Classification"
client = mlflow.tracking.MlflowClient()

if client.get_experiment_by_name(experiment_name) is None:
    client.create_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created on {MODE}")
else:
    print(f"Experiment '{experiment_name}' exists on {MODE}")

mlflow.set_experiment(experiment_name)

df = pd.read_csv("MLProject/text_emotion_preprocessing/text_emotion_clean.csv")
X = df["clean_text"].fillna("")
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():

    max_features = 5000
    max_iter = 1000

    # Params
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)

    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=max_iter)
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
