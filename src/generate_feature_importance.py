# src/generate_feature_importance.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# Resolve PROJECT ROOT correctly
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Paths (ABSOLUTE & CORRECT)
# -------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
PROC_TRAIN = BASE_DIR / "data" / "processed" / "train_processed.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("🔹 Loading trained Random Forest model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print("🔹 Loading processed training data...")
    df = pd.read_csv(PROC_TRAIN)

    X = df.drop(columns=["label"])

    # -------------------------------------------------
    # Feature importance
    # -------------------------------------------------
    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Save CSV
    csv_path = REPORT_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    print("📊 Feature importance saved at:", csv_path)

    # -------------------------------------------------
    # Plot top 20 features
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    importance_df.head(20).plot(
        kind="barh",
        x="Feature",
        y="Importance",
        legend=False
    )
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()

    plot_path = REPORT_DIR / "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    print("📈 Feature importance plot saved at:", plot_path)


if __name__ == "__main__":
    main()
