
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["category", "gender"]

NUMERIC_FEATURES = [
    "amount", "log_amount", "age", "city_pop", "distance_km",
    "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "amount_vs_median", "seconds_since_prev",
]

INDICATOR_FEATURES = [
    "ind_amount_vs_history", "ind_high_amount", "ind_night",
    "ind_risky_category", "ind_new_merchant", "ind_rapid_repeat",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = df["hour_of_day"].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def build_feature_matrix(df: pd.DataFrame):
    """Returns (X, y, feature_names, encoders)."""
    df = add_time_features(df)
    df, encoders = encode_categoricals(df)
    feature_cols = NUMERIC_FEATURES + INDICATOR_FEATURES + [f"{c}_enc" for c in CATEGORICAL_COLS]
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"] if "is_fraud" in df.columns else None
    return X, y, feature_cols, encoders


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    from fraud_labeling import build_labelled_dataset

    data = build_labelled_dataset(get_clean_dataset())
    X, y, cols, _ = build_feature_matrix(data)
    print("Feature matrix shape:", X.shape)
    print("Features:", cols)
    print("Fraud rate:", round(y.mean(), 4))
