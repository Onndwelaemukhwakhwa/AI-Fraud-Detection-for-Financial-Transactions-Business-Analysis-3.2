
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank_transactions_data_2.csv"

REQUIRED_COLUMNS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "PreviousTransactionDate",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Accept the transaction data from the CSV"""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and preprocess the data
    Drop duplicates of the data
    Parse the date columns to the datetime
    Delete/Remove rows with non-positive amounts/balance
    Remove whitespace from String"""

    df = df.copy()
    df = df.drop_duplicates(subset="TransactionID")

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")

    df = df.dropna(subset=["TransactionDate", "TransactionAmount", "AccountBalance"])
    df = df[df["TransactionAmount"] > 0]
    df = df[df["AccountBalance"] >= 0]

    for col in ["TransactionType", "Location", "Channel", "CustomerOccupation",
                "DeviceID", "MerchantID", "AccountID"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.sort_values("TransactionDate").reset_index(drop=True)
    return df


def get_clean_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience the entry points used"""
    return validate_and_clean(load_raw_data(path))


if __name__ == "__main__":
    data = get_clean_dataset()
    print(f"Loaded {len(data)} clean transactions across {data['AccountID'].nunique()} accounts.")
    print(data.head())
    print("\nDate range:", data["TransactionDate"].min(), "->", data["TransactionDate"].max())
