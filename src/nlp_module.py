

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RISK_PHRASES = [
    "night-time", "unusually large", "first time at this merchant",
    "far from the customer's home", "online purchase", "rapid repeat",
]


def _narrative(r) -> str:
    cat = r.category.replace("_", " ")
    text = (f"{cat.title()} purchase of {r.amount:,.2f} at {r.merchant} in {r.city}, {r.state} "
            f"by a {str(r.job).lower()} aged {int(r.age)}, at {r.hour_of_day:02d}:00.")
    if r.hour_of_day in (22, 23, 0, 1, 2, 3):
        text += " Night-time transaction."
    if r.amount_vs_median >= 5:
        text += f" Unusually large: {r.amount_vs_median:.0f} times this account's typical spend."
    if r.ind_new_merchant == 1:
        text += " First time at this merchant for this account."
    if r.distance_km > 120:
        text += f" Merchant is {r.distance_km:.0f} km from the customer's home, far from the customer's home area."
    if r.category.endswith("_net"):
        text += " Online purchase."
    if r.ind_rapid_repeat == 1:
        text += " Rapid repeat transaction within five minutes of the previous one."
    return text


def add_narratives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["narrative"] = [_narrative(r) for r in df.itertuples(index=False)]
    return df


def keyword_risk_score(text: str) -> int:
    text = text.lower()
    return sum(1 for p in RISK_PHRASES if p in text)


def add_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_narratives(df)
    df["text_risk_score"] = df["narrative"].apply(keyword_risk_score)
    return df


def build_tfidf_features(narratives: pd.Series, max_features: int = 300):
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vec.fit_transform(narratives), vec


def search_transactions(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    """Keyword-overlap search over narratives, highest fraud score first."""
    q = set(re.findall(r"[a-z0-9]+", query.lower())) - {"high", "risk", "show", "me", "transactions", "in", "the"}
    if not q:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
    low = df["narrative"].str.lower()
    match = sum(low.str.contains(r"\b" + re.escape(t) + r"\b", regex=True).astype(int) for t in q)
    scored = df.assign(_match=match)
    scored = scored[scored["_match"] > 0].sort_values(["_match", "fraud_risk_score"], ascending=[False, False])
    cols = ["transaction_id", "account_id", "city", "fraud_risk_score", "risk_category", "narrative"]
    return scored[cols].head(top_n)


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    from fraud_labeling import build_labelled_dataset
    from feature_engineering import add_time_features

    d = add_time_features(build_labelled_dataset(get_clean_dataset()))
    d = add_nlp_features(d)
    print(d[["transaction_id", "narrative", "text_risk_score"]].head(3).to_string(index=False))
    m, v = build_tfidf_features(d["narrative"])
    print("TF-IDF matrix:", m.shape)
    print("Text risk score vs fraud rate:")
    print(d.groupby("text_risk_score")["is_fraud"].mean().round(4))
