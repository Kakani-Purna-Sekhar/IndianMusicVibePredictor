"""
train_model.py
--------------
Loads the Spotify tracks dataset, trains a RandomForestClassifier,
evaluates it, and saves the model to model/model.pkl.

Run:  python train_model.py
"""

import os
import pickle
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "spotify_tracks.csv")
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "model.pkl")

# ── Feature columns used for training ────────────────────────────────────────
FEATURES = ["danceability", "energy", "acousticness", "valence"]
TARGET   = "artist_name"

# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and drop rows with missing feature values."""
    df = pd.read_csv(path)
    print(f"📂  Loaded {len(df)} rows from {path}")

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"🧹  Dropped {before - len(df)} rows with missing values. {len(df)} rows remain.")
    return df


def train(df: pd.DataFrame):
    """Train RandomForestClassifier and return (model, label_encoder, metrics)."""
    X = df[FEATURES].values
    y = df[TARGET].values

    # Encode string labels → integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 80 / 20 split, stratified so every class is represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    print(f"✂️   Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Random Forest – 200 trees, balanced class weights
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"\n🎯  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(report)

    metrics = {
        "accuracy": float(acc),
        "n_rows": int(len(df)),
        "n_artists": int(len(le.classes_)),
        "feature_columns": list(FEATURES),
        "model_type": type(clf).__name__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return clf, le, metrics


def save_model(clf, le, metrics: dict, path: str):
    """Pickle classifier, label encoder, and lightweight training metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": clf, "label_encoder": le, "metrics": metrics}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"💾  Model saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df        = load_and_clean(DATASET_PATH)
    clf, le, metrics = train(df)
    save_model(clf, le, metrics, MODEL_PATH)
    print("\n✅  Training complete! Run:  streamlit run app.py")
