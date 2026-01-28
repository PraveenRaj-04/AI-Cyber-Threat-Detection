# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# -------------------------------------------------
# Project root
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Correct dataset path (data/raw)
# -------------------------------------------------
RAW_FILE = BASE_DIR / "data" / "raw" / "nsl_kdd_dataset.csv"
PROC_DIR = BASE_DIR / "data" / "processed"

PROC_TRAIN = PROC_DIR / "train_processed.csv"
PROC_TEST = PROC_DIR / "test_processed.csv"


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at: {RAW_FILE}")

    print(f"✅ Dataset found: {RAW_FILE.name}")

    df = pd.read_csv(RAW_FILE)

    print("📊 Dataset shape:", df.shape)

    # Ensure label column
    if "label" not in df.columns:
        df.rename(columns={df.columns[-1]: "label"}, inplace=True)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    df_encoded = X.copy()
    df_encoded["label"] = y

    # Split
    train_df, test_df = train_test_split(
        df_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROC_TRAIN, index=False)
    test_df.to_csv(PROC_TEST, index=False)

    print("✅ Preprocessing completed successfully!")
    print("📁 Saved:", PROC_TRAIN)
    print("📁 Saved:", PROC_TEST)


if __name__ == "__main__":
    main()
