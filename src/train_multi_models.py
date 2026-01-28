# src/train_multi_models.py

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# Resolve PROJECT ROOT correctly
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Absolute paths (SAFE)
# -------------------------------------------------
PROC_TRAIN = BASE_DIR / "data" / "processed" / "train_processed.csv"
PROC_TEST  = BASE_DIR / "data" / "processed" / "test_processed.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("🔹 Loading processed dataset...")

    # -----------------------------
    # Check files exist
    # -----------------------------
    if not PROC_TRAIN.exists():
        raise FileNotFoundError(f"Train file not found: {PROC_TRAIN}")

    if not PROC_TEST.exists():
        raise FileNotFoundError(f"Test file not found: {PROC_TEST}")

    # -----------------------------
    # Load data
    # -----------------------------
    train_df = pd.read_csv(PROC_TRAIN)
    test_df = pd.read_csv(PROC_TEST)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # -----------------------------
    # Models to compare
    # -----------------------------
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    results = []

    # -----------------------------
    # Train & evaluate models
    # -----------------------------
    for name, model in models.items():
        print(f"🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        print(f"✅ {name} Accuracy: {acc * 100:.2f}%")

    # -----------------------------
    # Save comparison report
    # -----------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    report_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(report_path, index=False)

    print("📊 Model comparison saved at:", report_path)


if __name__ == "__main__":
    main()
