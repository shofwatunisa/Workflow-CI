import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# Load dataset
df = pd.read_csv("dataset/text_emotion_clean.csv")
df = df.dropna(subset=['clean_text', 'label_encoded'])

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf = RandomForestClassifier(random_state=args.random_state)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

# MLflow tracking
mlflow.set_experiment("TextEmotion_CI")

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Log metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

    # Artefak
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

    print(f"Best params: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
