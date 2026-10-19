

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    daily = (df.set_index("timestamp").resample("D")
             .agg(transaction_count=("transaction_id", "count"),
                  total_amount=("amount", "sum"),
                  fraud_count=("is_fraud", "sum")))
    daily["fraud_rate"] = daily["fraud_count"] / daily["transaction_count"].clip(lower=1)
    daily["rolling_7d_amount"] = daily["total_amount"].rolling(7, min_periods=1).mean()
    daily["rolling_7d_fraud_rate"] = daily["fraud_rate"].rolling(7, min_periods=1).mean()
    daily["rolling_7d_volatility"] = daily["total_amount"].rolling(7, min_periods=1).std()
    return daily


def naive_moving_average_forecast(daily: pd.DataFrame, horizon: int = 7, window: int = 7) -> pd.Series:
    last_avg = daily["total_amount"].tail(window).mean()
    idx = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series([last_avg] * horizon, index=idx, name="forecast_total_amount")


def hourly_fraud_pattern(df: pd.DataFrame) -> pd.Series:
    hours = df["timestamp"].dt.hour
    return df.groupby(hours)["is_fraud"].mean().sort_index()


# kept for dashboard/chatbot compatibility
def hourly_risk_pattern(df: pd.DataFrame) -> pd.Series:
    return hourly_fraud_pattern(df)


def plot_daily_trend(daily, forecast):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    ax[0].plot(daily.index, daily["total_amount"], alpha=0.4, label="Daily total amount")
    ax[0].plot(daily.index, daily["rolling_7d_amount"], linewidth=2, label="7-day rolling average")
    ax[0].plot(forecast.index, forecast.values, "r--", label="Naive 7-day forecast")
    ax[0].set_title("Daily Transaction Value, Rolling Average and Naive Forecast")
    ax[0].legend()
    ax[1].plot(daily.index, daily["rolling_7d_fraud_rate"] * 100, color="#d62728")
    ax[1].set_title("7-Day Rolling Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_series_trend.png", dpi=120)
    plt.close()


def plot_hourly(hourly):
    plt.figure(figsize=(9, 5))
    (hourly * 100).plot(kind="bar", color="#7c3aed")
    plt.title("Fraud Rate by Hour of Day (%)")
    plt.xlabel("Hour of day")
    plt.ylabel("% of transactions that are fraud")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hourly_fraud_pattern.png", dpi=120)
    plt.close()


def run_time_series_analysis(df: pd.DataFrame):
    daily = daily_aggregates(df)
    forecast = naive_moving_average_forecast(daily)
    hourly = hourly_fraud_pattern(df)
    plot_daily_trend(daily, forecast)
    plot_hourly(hourly)
    print(f"Time-series charts saved to {OUTPUT_DIR}")
    print("\nLast 3 days:\n", daily.tail(3)[["transaction_count", "total_amount", "fraud_count", "fraud_rate"]])
    print("\nNaive 7-day forecast of daily value:", round(float(forecast.iloc[0]), 2))
    print("\nHighest-fraud hours:\n", (hourly.sort_values(ascending=False).head(4) * 100).round(2))
    return daily, forecast, hourly


if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    run_time_series_analysis(get_clean_dataset())
