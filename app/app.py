# app/app.py
import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path
import json

MODEL_PATH = Path("models/intrusion_rf.pkl")
META_PATH = Path("models/model_meta.json")

# Attack type mapping (same as in training)
DOS_ATTACKS = [
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm"
]
PROBE_ATTACKS = [
    "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"
]
R2L_ATTACKS = [
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"
]
U2R_ATTACKS = [
    "buffer_overflow", "loadmodule", "rootkit", "perl",
    "sqlattack", "xterm", "ps", "httptunnel"
]


def map_label_to_attack_type(label: str) -> str:
    lbl = str(label).lower().strip()
    if lbl == "normal":
        return "normal"
    if lbl in DOS_ATTACKS:
        return "dos"
    if lbl in PROBE_ATTACKS:
        return "probe"
    if lbl in R2L_ATTACKS:
        return "r2l"
    if lbl in U2R_ATTACKS:
        return "u2r"
    return "other_attack"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    return load(MODEL_PATH)


def load_meta():
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            return json.load(f)
    return None


def main():
    st.set_page_config(
        page_title="AI Cyber Threat Detection",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ AI-Based Cyber Threat Detection (NSL-KDD)")
    st.write("Detect malicious network traffic using a Machine Learning model trained on the NSL-KDD dataset.")

    meta = load_meta()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        if meta:
            st.metric("Validation Accuracy", f"{meta['accuracy']*100:.2f}%")
            st.write("**Train Samples:**", meta["n_train_samples"])
            st.write("**Validation Samples:**", meta["n_val_samples"])
        else:
            st.write("Train the model to view accuracy and stats.")
        st.markdown("---")
        st.write("**Attack Types:**")
        st.write("- `normal` – Normal traffic")
        st.write("- `dos` – Denial of Service attack")
        st.write("- `probe` – Scanning / probing")
        st.write("- `r2l` – Remote to Local attack")
        st.write("- `u2r` – User to Root attack")

    model = load_model()

    uploaded_file = st.file_uploader("📂 Upload CSV file (processed features)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        # Separate features
        if "label" in df.columns:
            features = df.drop(columns=["label"])
        else:
            features = df

        # Predictions
        preds = model.predict(features)
        df["prediction"] = preds
        df["attack_type"] = df["prediction"].apply(map_label_to_attack_type)

        # Summary metrics
        total = len(df)
        normal_count = (df["attack_type"] == "normal").sum()
        attack_count = total - normal_count

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", total)
        with col2:
            st.metric("Normal Traffic", normal_count)
        with col3:
            st.metric("Suspicious / Attack", attack_count)

        st.subheader("📊 Attack Type Distribution")
        st.write(df["attack_type"].value_counts())

        st.subheader("🔍 Sample Predictions (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True)

        st.info("Tip: You can use this output as evidence in your project report to show detection of different attack categories.")

    else:
        st.info("Upload a processed CSV file (e.g., `data/processed/test_processed.csv`) to see predictions.")


if __name__ == "__main__":
    main()
