import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_synthetic_transactions(n_transactions: int = 25000) -> pd.DataFrame:
    customer_ids = np.random.randint(10000, 13000, size=n_transactions)
    merchant_ids = np.random.randint(500, 900, size=n_transactions)

    merchant_categories = np.random.choice(
        ["grocery", "fuel", "electronics", "travel", "fashion", "gaming", "crypto", "jewelry", "restaurant"],
        size=n_transactions,
        p=[0.22, 0.14, 0.12, 0.10, 0.13, 0.08, 0.05, 0.04, 0.12]
    )

    payment_methods = np.random.choice(
        ["credit_card", "debit_card", "wallet", "upi", "net_banking"],
        size=n_transactions,
        p=[0.34, 0.25, 0.14, 0.20, 0.07]
    )

    locations = np.random.choice(
        ["same_city", "same_state", "different_state", "international"],
        size=n_transactions,
        p=[0.58, 0.22, 0.15, 0.05]
    )

    hour = np.random.randint(0, 24, size=n_transactions)
    day_of_week = np.random.randint(0, 7, size=n_transactions)

    base_amount = np.random.lognormal(mean=4.7, sigma=0.9, size=n_transactions)
    category_multiplier = pd.Series(merchant_categories).map({
        "grocery": 0.7,
        "fuel": 0.8,
        "electronics": 2.2,
        "travel": 2.8,
        "fashion": 1.3,
        "gaming": 1.1,
        "crypto": 3.2,
        "jewelry": 4.0,
        "restaurant": 0.9
    }).values

    amount = base_amount * category_multiplier
    amount = np.round(np.clip(amount, 50, 75000), 2)

    failed_attempts_24h = np.random.poisson(lam=0.25, size=n_transactions)
    txn_count_1h = np.random.poisson(lam=1.1, size=n_transactions)
    txn_count_24h = txn_count_1h + np.random.poisson(lam=4.5, size=n_transactions)

    account_age_days = np.random.randint(1, 2500, size=n_transactions)
    previous_chargebacks = np.random.poisson(lam=0.08, size=n_transactions)
    device_changed = np.random.choice([0, 1], size=n_transactions, p=[0.86, 0.14])
    ip_location_mismatch = np.random.choice([0, 1], size=n_transactions, p=[0.88, 0.12])

    high_risk_category = np.isin(merchant_categories, ["crypto", "jewelry", "gaming"]).astype(int)
    night_txn = ((hour <= 5) | (hour >= 23)).astype(int)
    international = (locations == "international").astype(int)

    # Fraud probability rule with noise
    fraud_score = (
        0.000045 * amount
        + 0.32 * high_risk_category
        + 0.28 * night_txn
        + 0.35 * international
        + 0.22 * device_changed
        + 0.30 * ip_location_mismatch
        + 0.18 * failed_attempts_24h
        + 0.08 * txn_count_1h
        + 0.40 * (account_age_days < 60).astype(int)
        + 0.45 * (previous_chargebacks > 0).astype(int)
        + np.random.normal(0, 0.25, n_transactions)
    )

    fraud_probability = 1 / (1 + np.exp(-(fraud_score - 1.65)))
    is_fraud = np.random.binomial(1, fraud_probability)

    # Make fraud rare but meaningful
    rare_mask = np.random.rand(n_transactions) < 0.84
    is_fraud = np.where(rare_mask & (fraud_probability < 0.55), 0, is_fraud)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(1, n_transactions + 1)],
        "customer_id": customer_ids,
        "merchant_id": merchant_ids,
        "merchant_category": merchant_categories,
        "payment_method": payment_methods,
        "location_type": locations,
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
        "is_fraud": is_fraud
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_transactions()
    output_path = DATA_DIR / "synthetic_transactions.csv"
    df.to_csv(output_path, index=False)
    fraud_rate = df["is_fraud"].mean()
    print(f"Saved dataset to {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud rate: {fraud_rate:.2%}")
