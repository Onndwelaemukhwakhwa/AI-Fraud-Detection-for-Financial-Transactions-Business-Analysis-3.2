import re
import sys
from pathlib import Path

# Make "src" importable even when this file is run directly (e.g. from an IDE)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.nlp_module import search_transactions

HELP_TEXT = (
    "I'm the Fraud Investigation Assistant. I can help with:\n"
    "  - 'summary' - overall fraud-risk summary\n"
    "  - 'high risk' - the highest-risk transactions\n"
    "  - 'account <AccountID>' - risk profile for an account (e.g. account AC0042)\n"
    "  - 'search <keywords>' - search transaction narratives (e.g. 'search night-time online')\n"
    "  - 'riskiest hour' - the hour of day with the highest fraud rate\n"
    "  - 'riskiest category' - the merchant category with the highest fraud rate\n"
    "  - 'help' - show this message\n"
    "I only answer questions about the fraud-risk data in this prototype."
)


class FraudChatbot:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _summary(self) -> str:
        c = self.df["risk_category"].value_counts()
        flagged = self.df[self.df["risk_category"] == "High"]
        caught = int(flagged["is_fraud"].sum()) if "is_fraud" in self.df else 0
        total_fraud = int(self.df["is_fraud"].sum()) if "is_fraud" in self.df else 0
        return (f"{len(self.df):,} transactions analysed: {c.get('High', 0):,} High risk, "
                f"{c.get('Medium', 0):,} Medium, {c.get('Low', 0):,} Low. "
                f"The High-risk group contains {caught} of the {total_fraud} confirmed fraud cases.")

    def _high_risk(self, n=5) -> str:
        top = self.df[self.df["risk_category"] == "High"].sort_values("fraud_risk_score", ascending=False).head(n)
        if top.empty:
            return "No high-risk transactions found."
        lines = [f"  {r.transaction_id} | {r.account_id} | {r.city} | amount {r.amount:,.2f} | score {r.fraud_risk_score:.1f}"
                 for r in top.itertuples()]
        return f"Top {len(top)} highest-risk transactions:\n" + "\n".join(lines)

    def _account(self, acc_id: str) -> str:
        acc = self.df[self.df["account_id"].str.upper() == acc_id.upper()]
        if acc.empty:
            return f"I couldn't find account {acc_id.upper()} in this dataset."
        high = int((acc["risk_category"] == "High").sum())
        return (f"Account {acc_id.upper()}: {len(acc):,} transactions, {high} flagged High risk, "
                f"average risk score {acc['fraud_risk_score'].mean():.1f}/100, "
                f"highest score {acc['fraud_risk_score'].max():.1f}.")

    def _riskiest_hour(self) -> str:
        h = self.df.groupby(self.df["timestamp"].dt.hour)["is_fraud"].mean()
        return f"Fraud is most common at {h.idxmax()}:00, where {h.max():.1%} of transactions are fraud (overall rate {self.df['is_fraud'].mean():.1%})."

    def _riskiest_category(self) -> str:
        c = self.df.groupby("category")["is_fraud"].mean()
        return f"The riskiest merchant category is '{c.idxmax()}', with a fraud rate of {c.max():.1%}."

    def _search(self, q: str) -> str:
        res = search_transactions(self.df, q)
        if res.empty:
            return f"No transactions matched '{q}'."
        lines = [f"  {r.transaction_id} ({r.risk_category}, score {r.fraud_risk_score:.1f}) - {r.narrative}"
                 for r in res.itertuples()]
        return f"Top matches for '{q}':\n" + "\n".join(lines)

    def respond(self, message: str) -> str:
        t = message.strip().lower()
        if t in ("help", "?", ""):
            return HELP_TEXT
        if "summary" in t:
            return self._summary()
        if "riskiest hour" in t or "risky hour" in t:
            return self._riskiest_hour()
        if "riskiest category" in t:
            return self._riskiest_category()
        if "high risk" in t and not t.startswith("search"):
            return self._high_risk()
        m = re.search(r"account\s+([a-z0-9]+)", t)
        if m:
            return self._account(m.group(1))
        if t.startswith("search"):
            return self._search(t.replace("search", "", 1).strip())
        return "I'm not sure how to help with that within my scope. Type 'help' to see what I can answer."


if __name__ == "__main__":
    from src.pipeline import load_scored_data

    bot = FraudChatbot(load_scored_data())
    for q in ["summary", "high risk", "riskiest hour", "riskiest category", "account AC0001", "search night-time online"]:
        print(f"\n> {q}\n{bot.respond(q)}")


