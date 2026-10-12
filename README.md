# Transaction Categorizer for Balance

Classifies bank transaction descriptions into 17 detailed categories and maps
them to eight budget categories. The pipeline uses TF-IDF features and
calibrated logistic regression, then keyword rules when a prediction fails the
confidence or out-of-vocabulary (OOV) checks. Unresolved transactions return
`Needs Review`.

## Setup

```bash
git clone https://github.com/mvbs2005/transaction-categorizer.git
cd transaction-categorizer
git checkout exercise5-confidence-gate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Scikit-learn is pinned to 1.9.1 to match the supplied model files. The datasets
and trained models are included.

## Run

```bash
python scripts/demo.py --budgets
python scripts/demo.py --interactive
python scripts/evaluate.py
python -m pytest tests/ -q
```

To rebuild the synthetic datasets and default model:

```bash
python scripts/generate_data.py
python scripts/train.py
```

## Use from Python

```python
from src.categorize import categorize, categorize_batch, spending_by_budget_category

result = categorize("[debit] CHIPOTLE #1882", amount=14.80)
print(result.budget_category, result.confidence, result.needs_review)

transactions = [
    {"description": "[debit] CHIPOTLE #1882", "amount": 14.80, "direction": "debit"},
]
results = categorize_batch(transactions)
totals = spending_by_budget_category(transactions)
```

Each result includes the detailed category, budget category, confidence,
review flag, decision source, and alternative predictions. A rule-based result
retains the model's confidence value; that value is not a measured probability
for the rule's decision.

## Confidence gate evaluation

The tuning workflow reserves merchants and familiar descriptions from
`data/train.csv` for validation, searches both gate settings, and freezes the
selection before opening the final test files. It then refits on the original
training data and measures the selected settings on the final benchmarks.

```bash
python scripts/tune_gate.py
python scripts/demo.py --model-path models/categorizer_candidate.joblib --budgets
python scripts/evaluate.py --model-path models/categorizer_candidate.joblib
```

The candidate is exported to `models/categorizer_candidate.joblib`. To compare
it in another module, load it with `TransactionCategorizer.load()` and pass it
as the optional `model=` argument to the categorization functions.

The default model uses a confidence threshold of 0.55 and an OOV cutoff of
0.60. Validation selected 0.62 and 0.75 for the candidate. On the combined
final benchmark of 3,190 transactions:

| Measurement | Default settings | Candidate settings |
|---|---|---|
| Accepted model accuracy, including OOV check | 96.15% | 95.97% |
| Model gate review rate | 23.45% | 22.92% |
| Automatic accuracy after rule fallback | 96.43% | 96.11% |
| Review rate after rule fallback | 8.68% | 9.03% |

The candidate missed the model gate's 20% review target and reduced automatic
accuracy on unseen merchants. It remains optional; the original model is the
default. The evaluator measures the saved gate settings and never selects new
settings from the test-set sweep.

## Outputs

- `results/`: evaluation metrics, confusion matrix, probability-threshold
  diagnostics, and misclassified examples.
- `results/gate_tuning/`: joint search, validation split manifest, selected
  settings, final comparisons, changed decisions, and reliability plots with
  confidence-bin counts.
- `models/`: default and candidate models with JSON metadata. The candidate
  metadata records both gate settings and their validation provenance.

Both evaluation scripts accept `--output-dir`. The evaluator uses
`--model-path` to choose a saved model; the tuner uses it to set the candidate
export path. Running the scripts regenerates their outputs.

## Project layout

| Location | Purpose |
|---|---|
| `src/categorize.py` | Single and batch categorization, budget totals |
| `src/ml_model.py` | Normalization, features, training, saved gate settings |
| `src/gate_evaluation.py` | Validation splitting, joint search, reliability metrics |
| `src/categories.py`, `src/rules.py` | Category mapping and keyword fallback |
| `src/data_generator.py` | Synthetic transaction generation |
| `src/llm_classifier.py` | Optional Ollama comparison |
| `scripts/` | Data generation, training, evaluation, tuning, and demos |
| `tests/` | Categorization and confidence gate regression tests |

The optional local-LLM comparison requires Ollama with the requested model
available locally. Run it with `python scripts/evaluate.py --llm --llm-samples 100`.

## Limitations

Training data is synthetic, and unseen merchants are substantially harder
than familiar ones: raw model macro-F1 is 0.955 on familiar merchants and
0.382 on unseen merchants. Calibration does not guarantee reliable confidence
on new merchants. Categories are fixed, amounts are not model features, and
user corrections are not yet used for retraining. Plaid integration is future
work.

Original categorization pipeline: Srikar Mallajosyula. Validation and confidence
gate evaluation extension: Jackson Graff.
