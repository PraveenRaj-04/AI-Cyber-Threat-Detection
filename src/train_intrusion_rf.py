# src/train_intrusion_rf.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib

# -------------------------------------------------
# Resolve PROJECT ROOT correctly
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Paths (ABSOLUTE, SAFE)
# -------------------------------------------------
PROC_TRAIN = BASE_DIR / "data" / "processed" / "train_processed.csv"
PROC_TEST  = BASE_DIR / "data" / "processed" / "test_processed.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    print("🔹 Loading processed training data...")

    # -----------------------------
    # File existence check
    # -----------------------------
    if not PROC_TRAIN.exists():
        raise FileNotFoundError(f"Processed train file not found: {PROC_TRAIN}")

    if not PROC_TEST.exists():
        raise FileNotFoundError(f"Processed test file not found: {PROC_TEST}")

    # -----------------------------
    # Load data
    # -----------------------------
    train_df = pd.read_csv(PROC_TRAIN)
    test_df  = pd.read_csv(PROC_TEST)

    # -----------------------------
    # Split features & labels
    # -----------------------------
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # -----------------------------
    # Train Random Forest model
    # -----------------------------
    print("🔹 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate model
    # -----------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("✅ Training completed")
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Save model
    # -----------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
