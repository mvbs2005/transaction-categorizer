"""
Train the categorizer and save it to models/categorizer.joblib.

Run:  python scripts/train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml_model import DEFAULT_MODEL_PATH, TransactionCategorizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise SystemExit("data/train.csv is missing. Run: python scripts/generate_data.py")

    train_df = pd.read_csv(train_path)
    print(f"Training on {len(train_df)} transactions "
          f"across {train_df['source_category'].nunique()} categories")

    model = TransactionCategorizer()
    start = time.perf_counter()
    model.fit(train_df["description"].tolist(), train_df["source_category"].tolist())
    elapsed = time.perf_counter() - start

    path = model.save(DEFAULT_MODEL_PATH)
    size_kb = path.stat().st_size / 1024

    print(f"Trained in {elapsed:.1f}s")
    print(f"Word vocabulary   : {len(model.vocabulary_)} terms")
    print(f"Saved model       : {path.relative_to(ROOT)} ({size_kb:.0f} KB)")

    print("\nQuick sanity check (merchants the model trained on):")
    samples = [
        ("[debit] SQ *BLUE BOTTLE SAN FRANCISCO CA", "Coffee"),
        ("[debit] AMZN MKTP US*2K4L91", "Shopping"),
        ("[debit] ADOBE INC DES:BILLPAY ID:8831QK2LMN INDN:CARDHOLDER", "Subscriptions"),
        ("[credit] VENMO CASHOUT*8823QK", "Income"),
        ("[debit] VENMO CASHOUT*8823QK", "Transfer"),
    ]
    _show(model, samples)

    print("\nMerchants deliberately held out of training (the hard case):")
    unseen_samples = [
        ("[debit] STARBUCKS #4412 WEST LAFAYETTE IN", "Coffee"),
        ("[debit] WEGMANS #218 BROOKLYN NY", "Groceries"),
        ("[debit] REI.COM*K92LMQ4", "Shopping"),
    ]
    _show(model, unseen_samples)


def _show(model, samples: list[tuple[str, str]]) -> None:
    descriptions = [text for text, _ in samples]
    for prediction, (text, expected) in zip(model.predict_with_confidence(descriptions), samples):
        if prediction.uncertain:
            verdict = "REVIEW"
        elif prediction.source_category == expected:
            verdict = "ok"
        else:
            verdict = f"WRONG (expected {expected})"
        print(f"  {prediction.source_category:<14} {prediction.confidence:.2f}  {verdict:<26} {text}")


if __name__ == "__main__":
    main()
