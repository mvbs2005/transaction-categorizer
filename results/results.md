# Evaluation Results

All numbers produced by `python scripts/evaluate.py`.

## Approach comparison

| Approach | Test accuracy | Test macro-F1 | Unseen-merchant macro-F1 | Median latency |
|---|---|---|---|---|
| Rule baseline (21 rules) | 0.580 | 0.719 | 0.749 | <1 ms |
| TF-IDF + calibrated logistic regression | 0.962 | 0.955 | 0.382 | 1.82 ms |

The trained model improves macro-F1 over the rule baseline by **23.6 percentage points** on the same test set.
Trained model size: 1466.2 KB.

## Shipped pipeline (model -> rule fallback -> review queue)

| Slice | Auto-categorized | Accuracy when auto | Sent to review |
|---|---|---|---|
| Familiar merchants | 0.970 | 0.978 | 0.030 |
| New merchants | 0.768 | 0.920 | 0.232 |

## Confidence gate sweep (test + unseen merchants combined)

- Selected threshold: **0.55**
- Accuracy on accepted predictions: **0.949**
- Share flagged for user review: **0.202**

| Threshold | Coverage | Flagged for review | Accuracy on accepted |
|---|---|---|---|
| 0.20 | 0.998 | 0.002 | 0.798 |
| 0.25 | 0.981 | 0.019 | 0.812 |
| 0.30 | 0.934 | 0.066 | 0.848 |
| 0.35 | 0.887 | 0.113 | 0.886 |
| 0.40 | 0.855 | 0.145 | 0.912 |
| 0.45 | 0.835 | 0.165 | 0.926 |
| 0.50 | 0.824 | 0.176 | 0.931 |
| 0.55 | 0.798 | 0.202 | 0.949 |
| 0.60 | 0.781 | 0.219 | 0.955 |
| 0.65 | 0.754 | 0.245 | 0.969 |
| 0.70 | 0.735 | 0.265 | 0.972 |
| 0.75 | 0.718 | 0.282 | 0.981 |
| 0.80 | 0.706 | 0.294 | 0.981 |
| 0.85 | 0.692 | 0.308 | 0.984 |
| 0.90 | 0.683 | 0.317 | 0.987 |
| 0.95 | 0.646 | 0.354 | 0.993 |

## Most common confusions

| True | Predicted | Count |
|---|---|---|
| Shopping | Groceries | 38 |
| Groceries | Gas | 13 |
| Shopping | Healthcare | 13 |
| Shopping | Subscriptions | 12 |
| Groceries | Shopping | 6 |
| Gas | Groceries | 2 |

## Per-category performance (test set)

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Groceries | 0.726 | 0.848 | 0.782 | 125 |
| Dining | 1.000 | 1.000 | 1.000 | 134 |
| Coffee | 1.000 | 1.000 | 1.000 | 131 |
| Gas | 0.885 | 0.983 | 0.931 | 117 |
| Rideshare | 1.000 | 1.000 | 1.000 | 128 |
| Travel | 1.000 | 1.000 | 1.000 | 126 |
| Shopping | 0.892 | 0.468 | 0.614 | 124 |
| Subscriptions | 0.929 | 0.994 | 0.960 | 159 |
| Entertainment | 1.000 | 1.000 | 1.000 | 126 |
| Utilities | 1.000 | 1.000 | 1.000 | 163 |
| Rent | 1.000 | 1.000 | 1.000 | 160 |
| Healthcare | 0.905 | 1.000 | 0.950 | 124 |
| Fitness | 1.000 | 1.000 | 1.000 | 162 |
| Education | 0.994 | 1.000 | 0.997 | 159 |
| Income | 1.000 | 1.000 | 1.000 | 199 |
| Transfer | 1.000 | 1.000 | 1.000 | 102 |
| Fees | 1.000 | 1.000 | 1.000 | 57 |
