"""Train -> validate gate -> freeze settings -> refit -> evaluate final tests.

Run: python scripts/tune_gate.py
Only train.csv is used for fitting and choosing settings. The original test
files are opened after selection. This command exports an optional candidate,
including both settings and the validation provenance. Compare its results
before choosing it for the budgeting app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.categories import SOURCE_CATEGORIES
from src.gate_evaluation import (
    GatePredictions, joint_sweep, reliability_table, select_setting, validation_split,
)
from src.ml_model import (
    DEFAULT_MODEL_PATH, DEFAULT_OOV_CUTOFF, DEFAULT_THRESHOLD,
    TransactionCategorizer, acceptance_mask,
)

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_reliability(slices: dict[str, GatePredictions], output: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot([0, 1], [0, 1], "--", color="#88939c", label="Perfect calibration")
    metrics = {}
    for label, predictions in slices.items():
        table, ece = reliability_table(predictions.confidence, predictions.predicted == predictions.truth)
        table.to_csv(output.with_name(f"{output.stem}_{label}_bins.csv"), index=False)
        populated = table[table["count"] > 0]
        axes[0].plot(populated["mean_confidence"], populated["accuracy"], "o-", label=f"{label} (ECE {ece:.3f})")
        axes[1].step((table["lower"] + table["upper"]) / 2, table["count"], where="mid", label=label)
        metrics[label] = {"top_label_ece_10_bins": ece, "sample_count": len(predictions.truth)}
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean top-label confidence", ylabel="Observed accuracy")
    axes[1].set(xlim=(0, 1), xlabel="Confidence bin midpoint", ylabel="Transactions in bin")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
    fig.suptitle("Confidence reliability by merchant slice")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return metrics


def plot_search(sweep: pd.DataFrame, selected: dict, baseline: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    points = ax.scatter(sweep["review_rate"] * 100, sweep["accuracy_on_accepted"] * 100,
                        c=sweep["oov_cutoff"], cmap="viridis", s=10, alpha=0.45)
    ax.axhline(95, color="#70757a", ls="--", lw=1)
    ax.axvline(20, color="#70757a", ls="--", lw=1)
    for label, row, marker, color in [("Original settings", baseline, "s", "#dd7028"),
                                       ("Validation selection", selected, "*", "#ca304b")]:
        ax.scatter(row["review_rate"] * 100, row["accuracy_on_accepted"] * 100,
                   marker=marker, color=color, s=160, edgecolor="white", label=label, zorder=5)
    ax.set(xlabel="Sent to review before rule fallback (%)", ylabel="Accuracy on accepted model predictions (%)",
           title="Validation only | Joint confidence and OOV search")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    fig.colorbar(points, ax=ax, label="OOV cutoff")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def check_final_splits(training: pd.DataFrame, test: pd.DataFrame, unseen: pd.DataFrame) -> None:
    frames = [training, test, unseen]
    for index, left in enumerate(frames):
        for right in frames[index + 1:]:
            if set(left["description"]) & set(right["description"]):
                raise ValueError("Exact descriptions overlap between input datasets.")
    if set(training["merchant"]) & set(unseen["merchant"]):
        raise ValueError("Final unseen merchants overlap training merchants.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gate_tuning")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH.with_name("categorizer_candidate.joblib"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(args.data_dir / "train.csv")
    fitting, validation = validation_split(training, args.seed)
    manifest = training[["description", "merchant", "source_category"]].copy()
    manifest["description_sha256"] = manifest.pop("description").map(lambda d: hashlib.sha256(d.encode()).hexdigest())
    manifest["partition"] = "fitting"
    manifest.loc[validation.index, "partition"] = validation["validation_slice"].map("validation_{}".format)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    split = {"seed": args.seed, "original_training_rows": len(training), "fitting_rows": len(fitting),
             "validation_rows": len(validation),
             "validation_slices": {str(k): int(v) for k, v in validation["validation_slice"].value_counts().items()},
             "reserved_merchants": sorted(set(validation.loc[validation["validation_slice"] == "unseen", "merchant"])),
             "training_sha256": sha256(args.data_dir / "train.csv"),
             "split_manifest_sha256": sha256(out / "split_manifest.csv")}
    print(f"Training validation model on {len(fitting)} rows; validating on {len(validation)} rows.", flush=True)
    validation_model = TransactionCategorizer().fit(fitting["description"].tolist(), fitting["source_category"].tolist())
    validation_predictions = GatePredictions.from_frame(validation_model, validation)
    sweep = joint_sweep(validation_predictions)
    selected = select_setting(sweep)
    sweep.to_csv(out / "joint_sweep.csv", index=False)
    baseline_validation = validation_predictions.score(DEFAULT_THRESHOLD, DEFAULT_OOV_CUTOFF)
    reliability_validation = plot_reliability({label: GatePredictions.from_frame(validation_model, frame)
                                               for label, frame in validation.groupby("validation_slice")},
                                              out / "reliability_validation.png")
    plot_search(sweep, selected, baseline_validation, out / "validation_tradeoff.png")
    selection = {"source": "train.csv validation only", "chosen": selected, "split": split,
                 "candidate_count": len(sweep),
                 "refit": "Classifier refitted on all original training rows after settings were frozen."}
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selection locked: threshold={selected['threshold']:.2f}, OOV cutoff={selected['oov_cutoff']:.2f}; "
          f"{selected['status']}.", flush=True)

    # Test input and labels have not been opened during model or gate selection.
    model = TransactionCategorizer(selected["threshold"], selected["oov_cutoff"])
    model.fit(training["description"].tolist(), training["source_category"].tolist())
    model.gate_selection_ = selection
    test = pd.read_csv(args.data_dir / "test.csv")
    unseen = pd.read_csv(args.data_dir / "unseen_merchants.csv")
    check_final_splits(training, test, unseen)
    frames = {"familiar": test, "unseen": unseen, "mixed": pd.concat([test, unseen], ignore_index=True)}
    predictions = {label: GatePredictions.from_frame(model, frame) for label, frame in frames.items()}
    final, rows = {}, []
    for label, values in predictions.items():
        final[label] = {
            "ungated_accuracy": float(accuracy_score(values.truth, values.predicted)),
            "ungated_macro_f1": float(f1_score(values.truth, values.predicted, labels=SOURCE_CATEGORIES,
                                               average="macro", zero_division=0)),
        }
        for setting, threshold, cutoff in [("baseline", DEFAULT_THRESHOLD, DEFAULT_OOV_CUTOFF),
                                            ("tuned", model.threshold, model.oov_cutoff)]:
            for method, use_rules in [("gate", False), ("pipeline", True)]:
                score = values.score(threshold, cutoff, use_rules)
                score["objective_met"] = bool(score["accuracy_on_accepted"] is not None and
                                               score["accuracy_on_accepted"] >= 0.95 and score["review_rate"] <= 0.2)
                final[label][f"{setting}_{method}"] = score
                rows.append({"slice": label, "setting": setting, "method": method,
                             **{key: value for key, value in score.items() if key != "decided_by"}})
    pd.DataFrame(rows).to_csv(out / "final_comparison.csv", index=False)
    reliability_test = plot_reliability({k: v for k, v in predictions.items() if k != "mixed"},
                                        out / "reliability_test.png")
    mixed = predictions["mixed"]
    before = acceptance_mask(mixed.confidence, mixed.oov, DEFAULT_THRESHOLD, DEFAULT_OOV_CUTOFF)
    after = acceptance_mask(mixed.confidence, mixed.oov, model.threshold, model.oov_cutoff)
    changes = frames["mixed"][["description", "source_category", "merchant"]].copy()
    changes["prediction"] = mixed.predicted
    changes["confidence"] = mixed.confidence
    changes["oov_ratio"] = mixed.oov
    changes["baseline_accepted"] = before
    changes["tuned_accepted"] = after
    changes[before != after].to_csv(out / "changed_decisions.csv", index=False)
    metrics = {"author": "Jackson Graff", "date": datetime.now(ZoneInfo("America/Indianapolis")).isoformat(),
               "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
               "selection": selection, "baseline_validation_gate": baseline_validation,
               "validation_reliability": reliability_validation, "final": final,
               "final_reliability": reliability_test, "changed_model_acceptance_count": int((before != after).sum()),
               "final_input_sha256": {name: sha256(args.data_dir / name) for name in ["test.csv", "unseen_merchants.csv"]}}
    (out / "experiment.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    model.save(args.model_path)
    print("\nFinal evaluation (settings fixed before test labels were read)")
    print(f"{'Slice / decision chain':<28} {'Setting':<10} {'Accuracy':>10} {'Review':>10} {'Accepted':>10}")
    for row in rows:
        accuracy = row['accuracy_on_accepted']
        formatted = f"{accuracy:.2%}" if accuracy is not None else "n/a"
        print(f"{row['slice'] + ' / ' + row['method']:<28} {row['setting']:<10} "
              f"{formatted:>10} {row['review_rate']:>9.2%} {row['accepted_count']:>10}")
    print(f"\nSaved both gate settings and validation provenance to {args.model_path}.")
    print(f"Plots, sweep, split audit, and comparison written to {out}.")


if __name__ == "__main__":
    main()
