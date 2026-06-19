import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_transactions.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/generate_data.py")

    df = pd.read_csv(DATA_PATH)
    engineered = build_features(df)

    X = engineered.drop(columns=["is_fraud"])
    y = engineered["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Class imbalance handling through class_weight
    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    anomaly_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=200,
            contamination=max(0.01, min(0.12, y.mean())),
            random_state=42
        ))
    ])
    anomaly_pipeline.fit(X_train)

    metrics = {
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    joblib.dump(classifier, MODEL_DIR / "fraud_classifier.joblib")
    joblib.dump(anomaly_pipeline, MODEL_DIR / "anomaly_detector.joblib")

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(REPORT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Fraud rate: {metrics['fraud_rate']:.2%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Saved model files to {MODEL_DIR}")

if __name__ == "__main__":
    train()
