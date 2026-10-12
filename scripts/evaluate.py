"""
Evaluate rules, trained models, and the saved confidence gate.

Outputs written to results/:

    metrics.json          - every number in machine-readable form
    confusion_matrix.png  - where the model confuses categories
    threshold_sweep.csv   - accuracy vs coverage for the confidence gate
    error_samples.csv     - the actual mistakes, for error analysis

Run:  python scripts/evaluate.py
      python scripts/evaluate.py --llm          (adds the local-LLM arm)
      python scripts/evaluate.py --llm --llm-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm_classifier, rules  # noqa: E402
from src.categories import SOURCE_CATEGORIES, UNCERTAIN_LABEL  # noqa: E402
from src.ml_model import DEFAULT_MODEL_PATH, TransactionCategorizer  # noqa: E402
from src.gate_evaluation import GatePredictions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def score(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Accuracy and macro-F1, counting 'Needs Review' predictions as errors."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(y_true, y_pred, labels=SOURCE_CATEGORIES, average="macro", zero_division=0),
            4,
        ),
    }


def plot_confusion_matrix(y_true: list[str], y_pred: list[str], path: Path) -> None:
    labels = SOURCE_CATEGORIES
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted category", fontsize=11)
    ax.set_ylabel("True category", fontsize=11)
    ax.set_title("Confusion matrix — held-out test set (row-normalised)", fontsize=13, pad=14)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if value >= 0.01:
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value > 0.5 else "#222222",
                )

    fig.colorbar(image, ax=ax, fraction=0.045, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def threshold_sweep(
    model: TransactionCategorizer, descriptions: list[str], y_true: list[str]
) -> pd.DataFrame:
    """Accuracy on accepted predictions vs how many get accepted."""
    probabilities, classes = model._probabilities(descriptions)
    best_index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    predicted = np.array([classes[i] for i in best_index])
    truth = np.array(y_true)

    rows = []
    for threshold in np.arange(0.20, 0.96, 0.05):
        accepted = confidence >= threshold
        coverage = accepted.mean()
        accuracy = (predicted[accepted] == truth[accepted]).mean() if accepted.any() else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": round(float(coverage), 4),
                "flagged_for_review": round(float(1 - coverage), 4),
                "accuracy_on_accepted": round(float(accuracy), 4),
            }
        )
    return pd.DataFrame(rows)


def pipeline_score(model: TransactionCategorizer, frame: pd.DataFrame) -> dict:
    """Score the deployed decision chain: model -> rule fallback -> review.

    Three numbers matter here, and they trade off against each other:
      auto_rate          - share the system decides without asking the user
      accuracy_when_auto - how often those automatic decisions are right
      overall_accuracy   - counting anything sent to review as not-yet-correct
    """
    descriptions = frame["description"].tolist()
    directions = frame["direction"].tolist()
    truth = frame["source_category"].tolist()

    predictions = model.predict_with_confidence(descriptions)

    decided, correct_when_auto, auto_count = [], 0, 0
    decided_by = {"model": 0, "rules": 0, "review": 0}

    for prediction, description, direction in zip(predictions, descriptions, directions):
        if not prediction.uncertain:
            decided.append(prediction.source_category)
            decided_by["model"] += 1
        else:
            fallback = rules.classify(description, direction)
            decided.append(fallback)
            decided_by["rules" if fallback != UNCERTAIN_LABEL else "review"] += 1

    for label, actual in zip(decided, truth):
        if label != UNCERTAIN_LABEL:
            auto_count += 1
            correct_when_auto += int(label == actual)

    total = len(truth)
    return {
        "decided_by": decided_by,
        "auto_rate": round(auto_count / total, 4),
        "review_rate": round(1 - auto_count / total, 4),
        "accuracy_when_auto": round(correct_when_auto / auto_count, 4) if auto_count else 0.0,
        "overall_accuracy": round(correct_when_auto / total, 4),
    }


def measure_latency(model: TransactionCategorizer, descriptions: list[str], runs: int = 200) -> float:
    """Median single-transaction latency in milliseconds."""
    timings = []
    for description in descriptions[:runs]:
        start = time.perf_counter()
        model.predict_with_confidence([description])
        timings.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(timings)), 2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="include the local-LLM arm")
    parser.add_argument("--llm-samples", type=int, default=200)
    parser.add_argument("--llm-model", default=llm_classifier.DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    unseen_df = pd.read_csv(DATA_DIR / "unseen_merchants.csv")
    model = TransactionCategorizer.load(args.model_path)

    metrics: dict = {}

    test_descriptions = test_df["description"].tolist()
    test_truth = test_df["source_category"].tolist()
    unseen_descriptions = unseen_df["description"].tolist()
    unseen_truth = unseen_df["source_category"].tolist()

    # ---- 1. rule baseline ------------------------------------------------
    rule_test = rules.classify_many(test_descriptions, test_df["direction"].tolist())
    rule_unseen = rules.classify_many(unseen_descriptions, unseen_df["direction"].tolist())
    metrics["rules"] = {
        "rule_count": rules.rule_count(),
        # Precision and recall are reported separately because they tell the
        # real story: hand-written rules are very accurate when they fire and
        # simply stay silent the rest of the time.
        "macro_precision": round(
            precision_score(
                test_truth, rule_test, labels=SOURCE_CATEGORIES, average="macro", zero_division=0
            ),
            4,
        ),
        "macro_recall": round(
            recall_score(
                test_truth, rule_test, labels=SOURCE_CATEGORIES, average="macro", zero_division=0
            ),
            4,
        ),
        "test": score(test_truth, rule_test),
        "unseen_merchants": score(unseen_truth, rule_unseen),
        "unmatched_rate_test": round(rule_test.count(UNCERTAIN_LABEL) / len(rule_test), 4),
        "unmatched_rate_unseen": round(rule_unseen.count(UNCERTAIN_LABEL) / len(rule_unseen), 4),
    }

    # ---- 2. trained model ------------------------------------------------
    ml_test = model.predict(test_descriptions)
    ml_unseen = model.predict(unseen_descriptions)
    metrics["ml_model"] = {
        "test": score(test_truth, ml_test),
        "unseen_merchants": score(unseen_truth, ml_unseen),
        "median_latency_ms": measure_latency(model, test_descriptions),
        "model_size_kb": round(args.model_path.stat().st_size / 1024, 1),
    }

    # ---- 3. confidence gate ---------------------------------------------
    # The sweep is a diagnostic only. Choose parameters with tune_gate.py on
    # training-derived validation data, then measure the saved gate here.
    mixed_descriptions = test_descriptions + unseen_descriptions
    mixed_truth = test_truth + unseen_truth
    sweep = threshold_sweep(model, mixed_descriptions, mixed_truth)
    sweep.to_csv(results_dir / "threshold_sweep.csv", index=False)

    gate = GatePredictions.from_frame(model, pd.concat([test_df, unseen_df]))
    saved_gate = gate.score(model.threshold, model.oov_cutoff)
    accuracy = saved_gate["accuracy_on_accepted"]
    metrics["confidence_gate"] = {
        "objective_met": bool(accuracy is not None and accuracy >= 0.95 and saved_gate["review_rate"] <= 0.2),
        "selected_threshold": model.threshold,
        "oov_cutoff": model.oov_cutoff,
        "selection_source": model.gate_selection_.get("source", "legacy saved settings; not validation-tuned"),
        "accuracy_on_accepted": accuracy,
        "flagged_for_review": saved_gate["review_rate"],
        "accepted_count": saved_gate["accepted_count"],
    }

    # ---- 3b. the shipped pipeline ---------------------------------------
    # model -> (if uncertain) rules -> (if still unknown) flag for review.
    # This is what src/categorize.py actually does, so it is the number that
    # describes the deliverable rather than any single component.
    metrics["full_pipeline"] = {
        "threshold": model.threshold,
        "oov_cutoff": model.oov_cutoff,
        "test": pipeline_score(model, test_df),
        "unseen_merchants": pipeline_score(model, unseen_df),
    }

    # ---- 4. confusion matrix + error samples ----------------------------
    plot_confusion_matrix(test_truth, ml_test, results_dir / "confusion_matrix.png")

    errors = pd.DataFrame(
        {
            "description": test_descriptions,
            "true_category": test_truth,
            "predicted_category": ml_test,
            "merchant": test_df["merchant"],
        }
    )
    errors = errors[errors["true_category"] != errors["predicted_category"]]
    errors.head(60).to_csv(results_dir / "error_samples.csv", index=False)
    metrics["error_analysis"] = {
        "total_errors": int(len(errors)),
        "error_rate": round(len(errors) / len(test_df), 4),
        "top_confusions": [
            {"true": t, "predicted": p, "count": int(c)}
            for (t, p), c in errors.groupby(["true_category", "predicted_category"])
            .size()
            .sort_values(ascending=False)
            .head(6)
            .items()
        ],
    }

    report = classification_report(
        test_truth, ml_test, labels=SOURCE_CATEGORIES, zero_division=0, output_dict=True
    )
    metrics["per_category"] = {
        category: {
            "precision": round(values["precision"], 3),
            "recall": round(values["recall"], 3),
            "f1": round(values["f1-score"], 3),
            "support": int(values["support"]),
        }
        for category, values in report.items()
        if category in SOURCE_CATEGORIES
    }

    # ---- 5. optional local-LLM arm --------------------------------------
    if args.llm:
        available = llm_classifier.is_available(args.llm_model)
        if not available:
            print(f"Ollama not reachable — skipping the LLM arm ({args.llm_model}).")
            metrics["local_llm"] = {"available": False}
        else:
            sample = test_df.sample(n=min(args.llm_samples, len(test_df)), random_state=7)
            predictions, latencies, malformed = [], [], 0
            for _, row in sample.iterrows():
                result = llm_classifier.classify(
                    row["description"], row["amount"], model=args.llm_model
                )
                predictions.append(result.source_category)
                latencies.append(result.latency_ms)
                malformed += 0 if result.ok else 1
            metrics["local_llm"] = {
                "available": True,
                "model": args.llm_model,
                "sample_size": len(sample),
                **score(sample["source_category"].tolist(), predictions),
                "median_latency_ms": round(float(np.median(latencies)), 1),
                "malformed_response_rate": round(malformed / len(sample), 4),
            }
    else:
        metrics["local_llm"] = {"available": None, "note": "not run; pass --llm to include"}

    # ---- 6. write everything --------------------------------------------
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print_summary(metrics, results_dir)


def print_summary(metrics: dict, output_dir: Path = RESULTS_DIR) -> None:
    rb, ml, gate = metrics["rules"], metrics["ml_model"], metrics["confidence_gate"]
    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"{'':<34}{'accuracy':>10}{'macro-F1':>12}")
    print(f"{'Rule baseline (test)':<34}{rb['test']['accuracy']:>10.3f}{rb['test']['macro_f1']:>12.3f}")
    print(f"{'ML model (test)':<34}{ml['test']['accuracy']:>10.3f}{ml['test']['macro_f1']:>12.3f}")
    print(f"{'Rule baseline (unseen merchants)':<34}"
          f"{rb['unseen_merchants']['accuracy']:>10.3f}{rb['unseen_merchants']['macro_f1']:>12.3f}")
    print(f"{'ML model (unseen merchants)':<34}"
          f"{ml['unseen_merchants']['accuracy']:>10.3f}{ml['unseen_merchants']['macro_f1']:>12.3f}")
    print("-" * 62)
    gate_accuracy = f"{gate['accuracy_on_accepted']:.3f}" if gate['accuracy_on_accepted'] is not None else "n/a"
    print(f"Saved gate: threshold {gate['selected_threshold']}, OOV {gate['oov_cutoff']} -> "
          f"{gate_accuracy} accuracy on "
          f"{1 - gate['flagged_for_review']:.1%} of transactions")
    print(f"Median latency : {ml['median_latency_ms']} ms   Model size: {ml['model_size_kb']} KB")
    print(f"\nWrote metrics.json, confusion_matrix.png, threshold_sweep.csv, and error_samples.csv to {output_dir}")


if __name__ == "__main__":
    main()
