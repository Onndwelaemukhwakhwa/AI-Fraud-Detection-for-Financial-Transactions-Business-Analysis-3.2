
from pathlib import Path
import joblib
import pandas as pd

from src.data_preprocessing import get_clean_dataset
from src.fraud_labeling import build_labelled_dataset
from src.feature_engineering import add_time_features, build_feature_matrix
from src.ml_models import run_ml_pipeline, score_all_transactions, MODEL_DIR
from src.nlp_module import add_nlp_features

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "scored_transactions.pkl"


def build_scored_data(threshold: float = None) -> pd.DataFrame:
    data = build_labelled_dataset(get_clean_dataset())
    X, y, names, _ = build_feature_matrix(data)
    data = add_time_features(data)
    if threshold is None:
        model_file = MODEL_DIR / "random_forest_fraud_model.joblib"
        if model_file.exists():
            threshold = joblib.load(model_file)["threshold"]
        else:
            _, _, _, threshold = run_ml_pipeline(X, y, names, rule_score=data["rule_score"])
    scored = score_all_transactions(data, X, y, threshold)
    scored = add_nlp_features(scored)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_pickle(CACHE)
    return scored


def load_scored_data() -> pd.DataFrame:
    return pd.read_pickle(CACHE) if CACHE.exists() else build_scored_data()
