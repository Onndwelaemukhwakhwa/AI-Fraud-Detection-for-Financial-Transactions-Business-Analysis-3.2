
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df[["amount", "age", "city_pop", "distance_km"]].describe()


def plot_class_balance(df):
    plt.figure(figsize=(5, 4))
    counts = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#4c78a8", "#e45756"], legend=False)
    plt.yscale("log")
    plt.title("Class Balance (log scale)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=120); plt.close()


def plot_amount_by_class(df):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=np.log1p(df["amount"]), hue=df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}), common_norm=False)
    plt.title("Transaction Amount (log scale) by Class")
    plt.xlabel("log(1 + amount)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "amount_by_class.png", dpi=120); plt.close()


def plot_fraud_by_category(df):
    rate = (df.groupby("category")["is_fraud"].mean() * 100).sort_values()
    plt.figure(figsize=(8, 5))
    rate.plot(kind="barh", color="#e45756")
    plt.title("Fraud Rate by Merchant Category (%)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "fraud_by_category.png", dpi=120); plt.close()


def plot_correlation_heatmap(df):
    cols = ["amount", "age", "city_pop", "distance_km", "is_fraud"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation of Key Numeric Fields")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=120); plt.close()


def run_eda(df: pd.DataFrame):
    stats = summary_statistics(df)
    plot_class_balance(df); plot_amount_by_class(df)
    plot_fraud_by_category(df); plot_correlation_heatmap(df)
    stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
    print(f"EDA charts saved to {OUTPUT_DIR}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(df.groupby('is_fraud')['amount'].describe()[['mean', '50%', 'max']].round(2))
    return stats


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    print(run_eda(get_clean_dataset()))
