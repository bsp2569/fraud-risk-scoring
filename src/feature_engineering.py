import pandas as pd

HIGH_RISK_CATEGORIES = {"crypto", "jewelry", "gaming"}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["is_high_risk_category"] = data["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)
    data["is_night_transaction"] = ((data["hour"] <= 5) | (data["hour"] >= 23)).astype(int)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["is_new_account"] = (data["account_age_days"] < 60).astype(int)
    data["velocity_ratio"] = data["txn_count_1h"] / (data["txn_count_24h"] + 1)
    data["failed_attempt_rate"] = data["failed_attempts_24h"] / (data["txn_count_24h"] + 1)
    data["amount_log"] = (data["amount"] + 1).apply(lambda x: __import__("math").log(x))
    data["is_international"] = (data["location_type"] == "international").astype(int)

    categorical_cols = ["merchant_category", "payment_method", "location_type"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    drop_cols = ["transaction_id", "customer_id", "merchant_id"]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    return data
