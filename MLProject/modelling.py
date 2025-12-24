import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ==========================
# Set MLflow tracking URI lokal
# ==========================
mlruns_path = os.path.join(os.path.dirname(__file__), "mlruns")
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(mlruns_path)

# ==========================
# Set atau buat experiment
# ==========================
experiment_name = "TextEmotion_CI"
mlflow.set_experiment(experiment_name)

# ==========================
# Load dataset
# ==========================
dataset_path = os.path.join(os.path.dirname(__file__), "text_emotion_preprocessing", "text_emotion_clean.csv")
df = pd.read_csv(dataset_path)

X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Split data
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# Train model
# ==========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================
# Predict & metric
# ==========================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# ==========================
# Log params, metrics & model ke MLflow
# ==========================
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print(f"Training selesai, accuracy={acc}")
print(f"MLflow run tersimpan di {mlruns_path}")
