

from pathlib import Path

import pandas as pd

from src.data_preprocessing import get_clean_dataset
from src.fraud_labeling import build_labelled_dataset
from src.feature_engineering import add_time_features, build_feature_matrix
from src.nlp_module import add_nlp_features
from src.eda import run_eda
from src.time_series_analysis import run_time_series_analysis
from src.ml_models import run_ml_pipeline, score_all_transactions
from src.deep_learning_model import train_ann, train_autoencoder
from src.pipeline import CACHE
from src.chatbot import FraudChatbot


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def main():
    banner("STEP 1/7: Load and clean the transaction data")
    data = get_clean_dataset()
    print(f"{len(data):,} transactions, fraud rate {data['is_fraud'].mean():.2%}")

    banner("STEP 2/7: Risk indicators and rule-engine baseline")
    data = build_labelled_dataset(data)
    print(data["rule_category"].value_counts())

    banner("STEP 3/7: Feature engineering")
    X, y, names, _ = build_feature_matrix(data)
    data = add_time_features(data)
    print("Feature matrix:", X.shape)

    banner("STEP 4/7: Exploratory data analysis")
    run_eda(data)

    banner("STEP 5/7: Time-series analysis")
    run_time_series_analysis(data)

    banner("STEP 6/7: Machine learning and deep learning")
    ml_results, rf, scaler, thr = run_ml_pipeline(X, y, names, rule_score=data["rule_score"])
    _, _, ann_m = train_ann(X, y)
    _, _, ae_m = train_autoencoder(X, y)
    all_results = pd.concat([ml_results, pd.DataFrame([ann_m, ae_m])], ignore_index=True)
    print("\nFinal model comparison (test set):")
    print(all_results.to_string(index=False))
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    all_results.to_csv(out_dir / "model_comparison.csv", index=False)

    banner("STEP 7/7: Score every transaction, NLP narratives and chatbot demo")
    scored = add_nlp_features(score_all_transactions(data, X, y, thr))
    scored.to_pickle(CACHE)
    bot = FraudChatbot(scored)
    print(bot.respond("summary"))
    print(bot.respond("riskiest hour"))
    print("\nDone. Launch the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
