"""
evaluate_rf_updown.py
---------------------

Evaluate Random Forest regression models in an Up/Down classification setting
by comparing predicted next-day closes against previous closes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_CSV = Path("rf_updown_classification_report.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_stock_csv(stock_name: str) -> pd.DataFrame:
    """Load and preprocess a stock CSV."""
    csv_path = DATA_DIR / f"{stock_name}_NS.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{stock_name}: Missing required columns.")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return features matching the training pipeline."""
    return df[["Open", "High", "Low", "Volume"]]


def compute_updown_labels(
    close_series: pd.Series, prev_close: pd.Series
) -> List[str]:
    """Generate Up/Down labels comparing close price with previous close."""
    return np.where(close_series > prev_close, "Up", "Down")


def evaluate_stock(stock_name: str) -> Dict[str, object]:
    """Evaluate Random Forest model for a single stock."""
    df = load_stock_csv(stock_name)
    X = prepare_features(df)
    y = df["Close"]

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model_path = MODEL_DIR / f"{stock_name}_NS_RF_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"RF model not found for {stock_name}")

    model = joblib.load(model_path)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    prev_close = df["Close"].shift(1).iloc[split_idx:]
    mask = prev_close.notna()
    actual_labels = compute_updown_labels(y_test[mask], prev_close[mask])
    predicted_labels = compute_updown_labels(y_pred[mask], prev_close[mask])

    acc = accuracy_score(actual_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(actual_labels, predicted_labels, labels=["Up", "Down"])
    report_text = classification_report(
        actual_labels, predicted_labels, target_names=["Up", "Down"], digits=4
    )

    print(f"\n=== {stock_name} (Random Forest) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(pd.DataFrame(cm, index=["Up", "Down"], columns=["Up", "Down"]))
    print("Classification Report:")
    print(report_text)

    return {
        "Stock": stock_name,
        "Accuracy": acc,
        "Precision_weighted": precision,
        "Recall_weighted": recall,
        "F1_weighted": f1,
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    csv_files = [p for p in DATA_DIR.glob("*_NS.csv")]
    if not csv_files:
        raise FileNotFoundError("No stock CSV files found in data/.")

    stocks = [p.stem.replace("_NS", "") for p in csv_files]
    results = []
    for stock in stocks:
        try:
            results.append(evaluate_stock(stock))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Skipping {stock}: {exc}")

    if not results:
        print("❌ No stocks evaluated successfully.")
        return

    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv(REPORT_CSV, index=False)
    print("\nSummary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved summary to {REPORT_CSV.resolve()}")


if __name__ == "__main__":
    main()

