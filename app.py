import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import build_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_transactions.csv"
MODEL_PATH = ROOT / "models" / "fraud_classifier.joblib"
ANOMALY_PATH = ROOT / "models" / "anomaly_detector.joblib"
FEATURE_PATH = ROOT / "models" / "feature_columns.json"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

st.set_page_config(page_title="Fraud Risk Scoring Dashboard", layout="wide")

st.title("Fraud Risk Scoring & Transaction Anomaly Detection")
st.write("Monitor suspicious transactions, fraud-risk drivers, merchant risk patterns, and model-based investigation priorities.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_PATH)
    anomaly = joblib.load(ANOMALY_PATH)
    with open(FEATURE_PATH) as f:
        feature_cols = json.load(f)
    return clf, anomaly, feature_cols

def risk_band(score):
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Project files are missing. Run `python src/generate_data.py` and `python src/train_model.py` first.")
    st.stop()

df = load_data()
clf, anomaly_model, feature_cols = load_models()

features = build_features(df)
X = features.drop(columns=["is_fraud"])
X = X.reindex(columns=feature_cols, fill_value=0)

df["fraud_probability"] = clf.predict_proba(X)[:, 1]
df["risk_band"] = df["fraud_probability"].apply(risk_band)
df["anomaly_flag"] = anomaly_model.predict(X)
df["anomaly_flag"] = df["anomaly_flag"].map({-1: "Anomaly", 1: "Normal"})

with open(METRICS_PATH) as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Actual Fraud Rate", f"{df['is_fraud'].mean():.2%}")
col3.metric("High-Risk Transactions", f"{(df['risk_band'] == 'High').sum():,}")
col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Risk Score Distribution")
    fig = px.histogram(df, x="fraud_probability", nbins=40, title="Predicted Fraud Probability")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fraud by Merchant Category")
    fraud_by_cat = (
        df.groupby("merchant_category")
        .agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"), avg_risk=("fraud_probability", "mean"))
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    fig = px.bar(fraud_by_cat, x="merchant_category", y="avg_risk", title="Average Model Risk by Merchant Category")
    st.plotly_chart(fig, use_container_width=True)

left2, right2 = st.columns([1, 1])

with left2:
    st.subheader("Risk by Location Type")
    location_risk = (
        df.groupby("location_type")
        .agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"), avg_risk=("fraud_probability", "mean"))
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
    fig = px.bar(location_risk, x="location_type", y="avg_risk", title="Average Model Risk by Location")
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.subheader("Transaction Amount vs Risk")
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x="amount",
        y="fraud_probability",
        color="risk_band",
        hover_data=["transaction_id", "merchant_category", "payment_method", "location_type"],
        title="Amount vs Predicted Fraud Risk"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("High-Risk Investigation Queue")

risk_filter = st.selectbox("Select risk band", ["High", "Medium", "Low", "All"])
anomaly_filter = st.selectbox("Select anomaly flag", ["All", "Anomaly", "Normal"])

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_band"] == risk_filter]
if anomaly_filter != "All":
    filtered = filtered[filtered["anomaly_flag"] == anomaly_filter]

cols = [
    "transaction_id", "customer_id", "merchant_id", "merchant_category", "payment_method",
    "location_type", "amount", "hour", "failed_attempts_24h", "txn_count_1h",
    "device_changed", "ip_location_mismatch", "fraud_probability", "risk_band", "anomaly_flag", "is_fraud"
]

st.dataframe(
    filtered[cols].sort_values("fraud_probability", ascending=False).head(200),
    use_container_width=True
)

st.divider()

st.subheader("Single Transaction Risk Scoring")

with st.form("single_txn_form"):
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount", min_value=1.0, max_value=100000.0, value=5500.0)
    merchant_category = c2.selectbox("Merchant Category", sorted(df["merchant_category"].unique()))
    payment_method = c3.selectbox("Payment Method", sorted(df["payment_method"].unique()))

    c4, c5, c6 = st.columns(3)
    location_type = c4.selectbox("Location Type", sorted(df["location_type"].unique()))
    hour = c5.slider("Hour of Day", 0, 23, 2)
    day_of_week = c6.slider("Day of Week", 0, 6, 5)

    c7, c8, c9 = st.columns(3)
    failed_attempts_24h = c7.number_input("Failed Attempts 24h", min_value=0, max_value=20, value=2)
    txn_count_1h = c8.number_input("Txn Count 1h", min_value=0, max_value=50, value=4)
    txn_count_24h = c9.number_input("Txn Count 24h", min_value=1, max_value=100, value=12)

    c10, c11, c12 = st.columns(3)
    account_age_days = c10.number_input("Account Age Days", min_value=1, max_value=5000, value=45)
    previous_chargebacks = c11.number_input("Previous Chargebacks", min_value=0, max_value=20, value=1)
    device_changed = c12.selectbox("Device Changed", [0, 1])

    ip_location_mismatch = st.selectbox("IP Location Mismatch", [0, 1])

    submitted = st.form_submit_button("Score Transaction")

if submitted:
    new_row = pd.DataFrame([{
        "transaction_id": "MANUAL_TXN",
        "customer_id": 99999,
        "merchant_id": 999,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "location_type": location_type,
        "hour": hour,
        "day_of_week": day_of_week,
        "amount": amount,
        "failed_attempts_24h": failed_attempts_24h,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "account_age_days": account_age_days,
        "previous_chargebacks": previous_chargebacks,
        "device_changed": device_changed,
        "ip_location_mismatch": ip_location_mismatch,
        "is_fraud": 0
    }])

    new_features = build_features(new_row).drop(columns=["is_fraud"])
    new_features = new_features.reindex(columns=feature_cols, fill_value=0)

    score = clf.predict_proba(new_features)[0, 1]
    anomaly = anomaly_model.predict(new_features)[0]
    anomaly_label = "Anomaly" if anomaly == -1 else "Normal"

    st.success(f"Predicted fraud risk: {score:.2%} | Risk Band: {risk_band(score)} | Anomaly Flag: {anomaly_label}")
