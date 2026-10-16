import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import load_scored_data
from src.time_series_analysis import daily_aggregates, hourly_fraud_pattern
from src.chatbot import FraudChatbot

st.set_page_config(page_title="AI Fraud Risk Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading and scoring transactions (first run can take a few minutes)...")
def load_data():
    return load_scored_data()


def main():
    st.title("AI Fraud Risk Scoring Dashboard")
    st.caption("AI Solution for Industries - Financial Services | Business Analysis 3.2 (AIBUY3A). "
               "Decision-support tool: it ranks transactions for human review and does not make final fraud decisions.")
    df = load_data()

    st.sidebar.header("Filters")
    risk = st.sidebar.multiselect("Risk category", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    cats = st.sidebar.multiselect("Merchant category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    f = df[df["risk_category"].isin(risk) & df["category"].isin(cats)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(f):,}")
    c2.metric("Flagged High risk", f"{int((f['risk_category'] == 'High').sum()):,}")
    c3.metric("Confirmed fraud", f"{int(f['is_fraud'].sum()):,}")
    c4.metric("Accounts", f["account_id"].nunique())

    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Time Series", "Transaction Explorer", "Fraud Assistant", "Model Info"])

    with t1:
        a, b = st.columns(2)
        with a:
            st.subheader("Risk category distribution")
            st.bar_chart(f["risk_category"].value_counts().reindex(["Low", "Medium", "High"]))
        with b:
            st.subheader("Fraud rate by merchant category (%)")
            st.bar_chart((f.groupby("category")["is_fraud"].mean() * 100).sort_values())

    with t2:
        daily = daily_aggregates(f)
        st.subheader("Daily transaction value and 7-day rolling average")
        st.line_chart(daily[["total_amount", "rolling_7d_amount"]])
        st.subheader("Fraud rate by hour of day (%)")
        st.bar_chart(hourly_fraud_pattern(f) * 100)

    with t3:
        st.subheader("Highest-risk transactions")
        top = f[f["risk_category"] == "High"].sort_values("fraud_risk_score", ascending=False).head(500)
        st.dataframe(top[["transaction_id", "account_id", "city", "category", "amount", "fraud_risk_score", "is_fraud", "narrative"]],
                     width="stretch")
        acc = st.text_input("Look up an Account ID (e.g. AC0042)")
        if acc:
            st.dataframe(f[f["account_id"].str.upper() == acc.upper()].head(500))

    with t4:
        st.subheader("Ask the Fraud Investigation Assistant")
        bot = FraudChatbot(df)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        msg = st.chat_input("Try: summary, high risk, riskiest hour, account AC0042, search night-time online")
        if msg:
            st.session_state.chat_history += [("user", msg), ("assistant", bot.respond(msg))]
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.text(text)

    with t5:
        st.subheader("About this model")
        st.markdown(
            "- **Data**: simulated card transactions with a real fraud label (about 1% fraud), generated with the open-source Sparkov tool.\n"
            "- **Scoring**: a Random Forest gives each transaction a fraud probability. Every score is out-of-fold, so no transaction is scored by a model that trained on it.\n"
            "- **Compared against**: a rule-engine baseline, Logistic Regression, Isolation Forest, an ANN and an Autoencoder (see `src/ml_models.py` and `src/deep_learning_model.py`).\n"
            "- **Limitation**: the data is simulated, so real-world performance would differ."
        )


if __name__ == "__main__":
    main()
