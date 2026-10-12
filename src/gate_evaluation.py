"""Validation splits, joint gate selection, and top-label reliability diagnostics.

No function in this module reads the final test files. Gate statistics use the
same unrounded probability and OOV boundaries as production inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import rules
from .categories import UNCERTAIN_LABEL
from .ml_model import TransactionCategorizer, acceptance_mask


def validation_split(frame: pd.DataFrame, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve one merchant per label, then 25% of remaining rows as familiar validation.

    All rows for a reserved merchant stay out of fitting, even if that merchant
    spans multiple categories. Familiar rows share merchants but not exact
    descriptions with fitting rows, as in the original familiar test.
    """
    required = {"merchant", "description", "source_category", "direction"}
    if not required <= set(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Training data needs non-null merchant, description, category and direction.")
    if frame["description"].duplicated().any():
        raise ValueError("Remove duplicate descriptions before splitting.")
    rng = np.random.default_rng(seed)
    reserved: set[str] = set()
    labels = sorted(frame["source_category"].unique())
    for label in labels:
        candidates = sorted(frame.loc[frame["source_category"] == label, "merchant"].unique())
        if len(candidates) < 2:
            raise ValueError(f"Need at least two merchants for {label}.")
        for candidate in rng.permutation(candidates):
            proposed = reserved | {str(candidate)}
            remaining = frame.loc[~frame["merchant"].isin(proposed), "source_category"].value_counts()
            # Three calibration folds and a familiar validation split need
            # enough fitting examples of every class.
            if all(remaining.get(category, 0) >= 8 for category in labels):
                reserved = proposed
                break
        else:
            raise ValueError(f"Cannot hold out a merchant for {label} while retaining all classes.")
    unseen = frame.loc[frame["merchant"].isin(reserved)].copy()
    seen = frame.loc[~frame["merchant"].isin(reserved)].copy()
    fitting, familiar = train_test_split(
        seen, test_size=0.25, random_state=seed, stratify=seen["source_category"]
    )
    fitting, familiar = fitting.copy(), familiar.copy()
    familiar["validation_slice"] = "familiar"
    unseen["validation_slice"] = "unseen"
    validation = pd.concat([familiar, unseen]).sort_index()
    if not set(familiar["merchant"]) <= set(fitting["merchant"]):
        raise ValueError("A familiar validation merchant has no fitting examples.")
    if set(fitting["merchant"]) & set(unseen["merchant"]):
        raise AssertionError("Reserved merchants leaked into fitting.")
    if set(fitting["description"]) & set(validation["description"]):
        raise AssertionError("Duplicate descriptions leaked into validation.")
    return fitting.sort_index(), validation


@dataclass
class GatePredictions:
    predicted: np.ndarray
    truth: np.ndarray
    confidence: np.ndarray
    oov: np.ndarray
    fallback: np.ndarray

    @classmethod
    def from_frame(cls, model: TransactionCategorizer, frame: pd.DataFrame) -> "GatePredictions":
        if frame.empty:
            raise ValueError("Cannot evaluate an empty dataset.")
        descriptions = frame["description"].tolist()
        probabilities, classes = model._probabilities(descriptions)
        return cls(
            predicted=classes[probabilities.argmax(axis=1)],
            truth=frame["source_category"].to_numpy(),
            confidence=probabilities.max(axis=1),
            oov=np.array([model.oov_ratio(d) for d in descriptions]),
            fallback=np.array(rules.classify_many(descriptions, frame["direction"].tolist())),
        )

    def score(self, threshold: float, oov_cutoff: float, use_rules: bool = False) -> dict:
        accepted = acceptance_mask(self.confidence, self.oov, threshold, oov_cutoff)
        decided = np.where(accepted, self.predicted, self.fallback if use_rules else UNCERTAIN_LABEL)
        auto = decided != UNCERTAIN_LABEL
        total, count = len(auto), int(auto.sum())
        correct = int(((decided == self.truth) & auto).sum())
        metrics = {
            "total": total,
            "accepted_count": count,
            "correct_accepted": correct,
            "coverage": count / total,
            "review_rate": 1 - count / total,
            "accuracy_on_accepted": correct / count if count else None,
        }
        if use_rules:
            metrics["decided_by"] = {
                "model": int(accepted.sum()),
                "rules": int((~accepted & auto).sum()),
                "review": int((~auto).sum()),
            }
        return metrics


def joint_sweep(
    predictions: GatePredictions, thresholds=None, oov_cutoffs=None
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.2, 1.0, 0.01), 2) if thresholds is None else thresholds
    # Retain a safety check for descriptions with almost no familiar words.
    oov_cutoffs = np.round(np.arange(0.2, 0.91, 0.05), 2) if oov_cutoffs is None else oov_cutoffs
    rows = []
    for threshold in thresholds:
        for cutoff in oov_cutoffs:
            if not 0 <= threshold <= 1 or not 0 <= cutoff <= 1:
                raise ValueError("Gate settings must be between 0 and 1.")
            rows.append({"threshold": float(threshold), "oov_cutoff": float(cutoff),
                         **predictions.score(threshold, cutoff)})
    if not rows:
        raise ValueError("Provide at least one threshold and OOV cutoff.")
    return pd.DataFrame(rows)


def select_setting(sweep: pd.DataFrame, target_accuracy=0.95, max_review_rate=0.2) -> dict:
    """Maximize coverage under both limits; explicitly report infeasible targets."""
    if not 0 < target_accuracy <= 1 or not 0 <= max_review_rate < 1:
        raise ValueError("Invalid accuracy or review target.")
    usable = sweep[sweep["accepted_count"] > 0]
    if usable.empty:
        raise ValueError("No setting accepts a prediction.")
    accurate = usable[usable["accuracy_on_accepted"] >= target_accuracy]
    feasible = accurate[accurate["review_rate"] <= max_review_rate]
    if not feasible.empty:
        pool, status = feasible, "both_targets_met"
    elif not accurate.empty:
        pool, status = accurate, "review_target_unmet"
    else:
        pool, status = usable, "accuracy_target_unmet"
    ordering = ["coverage", "accuracy_on_accepted", "oov_cutoff", "threshold"]
    ascending = [False, False, True, False]
    if status == "accuracy_target_unmet":
        ordering[:2] = ["accuracy_on_accepted", "coverage"]
    chosen = pool.sort_values(ordering, ascending=ascending).iloc[0].to_dict()
    for key in ["total", "accepted_count", "correct_accepted"]:
        chosen[key] = int(chosen[key])
    return {**chosen, "objective_met": status == "both_targets_met", "status": status,
            "target_accuracy": target_accuracy, "max_review_rate": max_review_rate}


def reliability_table(confidence, correct, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Top-label confidence versus correctness; retain empty bins and sample counts.

    This is a multiclass top-label diagram, not a binary class calibration curve.
    ECE is the count-weighted absolute confidence/accuracy gap for these bins.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct)
    if confidence.ndim != 1 or correct.shape != confidence.shape or not len(confidence):
        raise ValueError("Provide matching, nonempty 1D arrays.")
    if bins < 1 or not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be finite and between 0 and 1; bins must be positive.")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("Correctness must contain booleans or 0/1.")
    assignment = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = assignment == index
        count = int(selected.sum())
        mean = float(confidence[selected].mean()) if count else np.nan
        accuracy = float(correct[selected].mean()) if count else np.nan
        if count:
            ece += count / len(confidence) * abs(mean - accuracy)
        rows.append({"lower": index / bins, "upper": (index + 1) / bins,
                     "count": count, "mean_confidence": mean, "accuracy": accuracy})
    return pd.DataFrame(rows), float(ece)
