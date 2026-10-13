
import numpy as np
import pandas as pd

# Categories with the highest fraud rates in the exploratory analysis (EDA).
RISKY_CATEGORIES = {"shopping_net", "misc_net", "grocery_pos", "shopping_pos"}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

INDICATOR_WEIGHTS = {
    "ind_amount_vs_history": 0.30,
    "ind_high_amount": 0.20,
    "ind_night": 0.20,
    "ind_risky_category": 0.15,
    "ind_new_merchant": 0.10,
    "ind_rapid_repeat": 0.05,
}


def add_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    # Amount compared with the account's own typical spend
    acc_median = df.groupby("account_id")["amount"].transform("median")
    df["amount_vs_median"] = df["amount"] / (acc_median + 1e-6)
    df["ind_amount_vs_history"] = np.clip((df["amount_vs_median"] - 1) / 9, 0, 1)

    # Large absolute amount
    p90 = df["amount"].quantile(0.90)
    df["ind_high_amount"] = np.clip(df["amount"] / (p90 * 3), 0, 1)

    # Night-time transaction
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["ind_night"] = df["hour_of_day"].isin(NIGHT_HOURS).astype(float)

    # Higher-risk merchant category
    df["ind_risky_category"] = df["category"].isin(RISKY_CATEGORIES).astype(float)

    # First time this account uses this merchant
    df["ind_new_merchant"] = (df.groupby(["account_id", "merchant"]).cumcount() == 0).astype(float)

    # Rapid repeat: previous transaction on the account under 5 minutes ago
    df["seconds_since_prev"] = df.groupby("account_id")["timestamp"].diff().dt.total_seconds()
    df["seconds_since_prev"] = df["seconds_since_prev"].fillna(df["seconds_since_prev"].median())
    df["ind_rapid_repeat"] = (df["seconds_since_prev"] < 300).astype(float)
    return df


def compute_rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted rule-engine score, 0-100."""
    df = add_risk_indicators(df)
    score = sum(df[col] * w for col, w in INDICATOR_WEIGHTS.items())
    df["rule_score"] = (score * 100).round(2)
    return df


def assign_risk_category(df: pd.DataFrame, score_col: str, medium: float, high: float,
                         out_col: str = "risk_category") -> pd.DataFrame:
    """FR06: Low / Medium / High from a score column and two cut-offs."""
    df = df.copy()
    df[out_col] = np.where(df[score_col] >= high, "High",
                           np.where(df[score_col] >= medium, "Medium", "Low"))
    return df


def build_labelled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Adds indicators and the rule-engine baseline score/category (top 5% = High)."""
    df = compute_rule_score(df)
    medium = df["rule_score"].quantile(0.85)
    high = df["rule_score"].quantile(0.95)
    return assign_risk_category(df, "rule_score", medium, high, out_col="rule_category")


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    from sklearn.metrics import roc_auc_score

    d = build_labelled_dataset(get_clean_dataset())
    print(d["rule_category"].value_counts())
    print("Rule-engine ROC-AUC vs real fraud label:", round(roc_auc_score(d["is_fraud"], d["rule_score"]), 4))
