import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description="Text Emotion Classification Training")
parser.add_argument(
    "--dataset_path",
    type=str,
    default="MLProject/text_emotion_preprocessing/text_emotion_clean.csv",  
    required=True,
    help="Path to the cleaned dataset CSV file"
)
parser.add_argument(
    "--target_column",
    type=str,
    default="clean_text",
    help="Name of the target column in dataset",
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
args = parser.parse_args()

dataset_path = args.dataset_path
target_column = args.target_column
random_state = args.random_state

# ----------------------------
# Load dataset
# ----------------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

df = pd.read_csv(dataset_path)

# ----------------------------
# Handle missing values
# ----------------------------
if df.isna().sum().sum() > 0:
    print("Found missing values. Dropping rows with NaN...")
    df = df.dropna(subset=[target_column, 'text'])  # pastikan kolom text dan target tidak NaN

# ----------------------------
# Check target column
# ----------------------------
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found in dataset columns.")

# ----------------------------
# Prepare features & labels
# ----------------------------
X = df['text']
y = df[target_column]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

# ----------------------------
# Feature extraction
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=500, random_state=random_state)
model.fit(X_train_tfidf, y_train)

# ----------------------------
# Evaluate model
# ----------------------------
y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# ----------------------------
# Save artifacts
# ----------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/text_emotion_model.pkl")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl")

print("Training complete. Model and vectorizer saved in 'artifacts/' folder.")
