import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df[["TransactionAmount", "CustomerAge", "TransactionDuration",
               "LoginAttempts", "AccountBalance"]].describe()


def plot_amount_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["TransactionAmount"], bins=40, kde=True, color="#2563eb")
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Transaction Amount")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=120)
    plt.close()


def plot_risk_category_counts(df: pd.DataFrame):
    plt.figure(figsize=(6, 5))
    order = ["Low", "Medium", "High"]
    sns.countplot(x="risk_category", hue="risk_category", data=df, order=order,
                  palette=["#22c55e", "#f59e0b", "#ef4444"], legend=False)
    plt.title("Transactions by Fraud Risk Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_category_counts.png", dpi=120)
    plt.close()


def plot_channel_vs_risk(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    cross = pd.crosstab(df["Channel"], df["risk_category"], normalize="index")
    cross = cross.reindex(columns=["Low", "Medium", "High"])
    cross.plot(kind="bar", stacked=True, color=["#22c55e", "#f59e0b", "#ef4444"])
    plt.title("Risk Category Proportion by Channel")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "channel_vs_risk.png", dpi=120)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()


def run_eda(df: pd.DataFrame):
    stats = summary_statistics(df)
    plot_amount_distribution(df)
    plot_risk_category_counts(df)
    plot_channel_vs_risk(df)
    plot_correlation_heatmap(df)
    stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
    print(f"EDA charts and statistics saved to {OUTPUT_DIR}")
    return stats


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    from fraud_labeling import build_labelled_dataset

    data = build_labelled_dataset(get_clean_dataset())
    print(run_eda(data))
