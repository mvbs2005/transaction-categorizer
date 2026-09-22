"""
Build the datasets.

Produces three files in data/:

    train.csv             - model training data (seen merchants only)
    test.csv              - held-out rows from the same merchants
    unseen_merchants.csv  - rows built ONLY from merchants withheld from
                            training, which is the honest generalisation test

Run:  python scripts/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import (  # noqa: E402
    deduplicate,
    generate,
    split_merchant_catalogue,
    to_records,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ROWS_PER_CATEGORY = 900
UNSEEN_ROWS_PER_CATEGORY = 60
TEST_FRACTION = 0.2
SEED = 42


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    seen_catalogue, held_out_catalogue = split_merchant_catalogue(
        holdout_fraction=0.25, seed=13
    )

    print("Merchant catalogue split")
    print(f"  training merchants : {sum(len(v) for v in seen_catalogue.values())}")
    print(f"  held-out merchants : {sum(len(v) for v in held_out_catalogue.values())}")

    # ---- main pool (seen merchants) -------------------------------------
    main_rows = generate(
        seen_catalogue,
        rows_per_category=ROWS_PER_CATEGORY,
        seed=SEED,
        merchant_seen_in_training=True,
        include_ambiguous=True,
    )
    before = len(main_rows)
    main_rows = deduplicate(main_rows)
    print(f"\nGenerated {before} rows, {before - len(main_rows)} exact duplicates removed")

    frame = pd.DataFrame(to_records(main_rows))

    train_df, test_df = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=frame["source_category"],
    )

    # ---- unseen-merchant slice ------------------------------------------
    unseen_rows = generate(
        held_out_catalogue,
        rows_per_category=UNSEEN_ROWS_PER_CATEGORY,
        seed=SEED + 1,
        merchant_seen_in_training=False,
        include_ambiguous=False,
    )
    unseen_rows = deduplicate(unseen_rows)
    unseen_df = pd.DataFrame(to_records(unseen_rows))

    # Safety check: no merchant may appear in both training and the unseen slice.
    overlap = set(train_df["merchant"]) & set(unseen_df["merchant"])
    if overlap:
        raise AssertionError(f"Merchant leakage into the unseen slice: {sorted(overlap)}")

    # Safety check: no identical description on both sides of the split.
    string_overlap = set(train_df["description"]) & set(test_df["description"])
    if string_overlap:
        raise AssertionError(f"{len(string_overlap)} duplicate descriptions leaked into test")

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    unseen_df.to_csv(DATA_DIR / "unseen_merchants.csv", index=False)

    print(f"\ntrain.csv            : {len(train_df):>6} rows")
    print(f"test.csv             : {len(test_df):>6} rows")
    print(f"unseen_merchants.csv : {len(unseen_df):>6} rows")
    print("\nLeakage checks passed: no shared merchants, no shared description strings.")
    print("\nSample rows:")
    for _, row in train_df.head(6).iterrows():
        print(f"  {row['source_category']:<14} | {row['description']}")


if __name__ == "__main__":
    main()
