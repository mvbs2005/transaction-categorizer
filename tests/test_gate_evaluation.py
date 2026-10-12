"""Tests for validation isolation, gate boundaries, selection, and reliability."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.categories import UNCERTAIN_LABEL
from src.gate_evaluation import (
    GatePredictions, joint_sweep, reliability_table, select_setting, validation_split,
)
from src.ml_model import TransactionCategorizer, acceptance_mask


def sample_training():
    return pd.DataFrame([
        {"merchant": f"{label} merchant {merchant}", "description": f"{label} {merchant} {row}",
         "source_category": label, "direction": "debit"}
        for label in ["Coffee", "Dining"] for merchant in range(4) for row in range(20)
    ])


def test_validation_partition_is_reproducible_and_has_no_merchant_leakage():
    data = sample_training()
    fitting, validation = validation_split(data)
    same_fit, same_validation = validation_split(data)
    pd.testing.assert_frame_equal(fitting, same_fit)
    pd.testing.assert_frame_equal(validation, same_validation)
    unseen = validation[validation["validation_slice"] == "unseen"]
    familiar = validation[validation["validation_slice"] == "familiar"]
    assert not set(fitting["merchant"]) & set(unseen["merchant"])
    assert set(familiar["merchant"]) <= set(fitting["merchant"])
    assert not set(fitting["description"]) & set(validation["description"])
    assert set(fitting.index) | set(validation.index) == set(data.index)
    assert set(fitting["source_category"]) == set(data["source_category"])


def test_duplicate_descriptions_cannot_enter_validation():
    data = sample_training()
    with pytest.raises(ValueError, match="duplicate"):
        validation_split(pd.concat([data, data.iloc[:1]], ignore_index=True))


def test_single_merchant_class_cannot_be_split_safely():
    data = sample_training()
    data["merchant"] = data["source_category"]
    with pytest.raises(ValueError, match="at least two merchants"):
        validation_split(data)


def test_gate_boundaries_use_unrounded_values():
    assert acceptance_mask([0.55, 0.54996, 0.95], [0.6, 0, 0.60001], 0.55, 0.6).tolist() == [True, False, False]


def test_production_gate_uses_configurable_oov_cutoff():
    model = TransactionCategorizer(0.55, 0.4)
    model._probabilities = lambda descriptions: (np.array([[0.55, 0.45], [0.96, 0.04]]), np.array(["Coffee", "Dining"]))
    model.oov_ratio = lambda description: {"known": 0.4, "unknown": 0.41}[description]
    result = model.predict_with_confidence(["known", "unknown"])
    assert not result[0].uncertain
    assert result[1].source_category == UNCERTAIN_LABEL
    assert result[1].reason == "unknown_merchant"


def test_gate_and_rule_fallback_count_actual_decisions():
    values = GatePredictions(
        predicted=np.array(["Coffee", "Shopping", "Shopping"]), truth=np.array(["Coffee", "Dining", "Coffee"]),
        confidence=np.array([0.95, 0.3, 0.9]), oov=np.array([0.1, 0.1, 0.9]),
        fallback=np.array([UNCERTAIN_LABEL, "Dining", UNCERTAIN_LABEL]),
    )
    gate = values.score(0.55, 0.6)
    pipeline = values.score(0.55, 0.6, use_rules=True)
    assert gate["accepted_count"] == 1 and gate["correct_accepted"] == 1
    assert pipeline["accepted_count"] == 2 and pipeline["correct_accepted"] == 2
    assert pipeline["decided_by"] == {"model": 1, "rules": 1, "review": 1}
    assert values.score(1, 0)["accuracy_on_accepted"] is None


def selection_rows():
    return pd.DataFrame([
        {"threshold": .5, "oov_cutoff": .6, "total": 100, "accepted_count": 90,
         "correct_accepted": 81, "coverage": .9, "review_rate": .1, "accuracy_on_accepted": .9},
        {"threshold": .6, "oov_cutoff": .6, "total": 100, "accepted_count": 85,
         "correct_accepted": 82, "coverage": .85, "review_rate": .15, "accuracy_on_accepted": 82 / 85},
        {"threshold": .7, "oov_cutoff": .6, "total": 100, "accepted_count": 70,
         "correct_accepted": 70, "coverage": .7, "review_rate": .3, "accuracy_on_accepted": 1},
    ])


def test_selection_maximizes_coverage_under_both_constraints():
    chosen = select_setting(selection_rows())
    assert chosen["threshold"] == .6 and chosen["objective_met"]


def test_selection_reports_infeasible_review_target():
    chosen = select_setting(selection_rows().iloc[[0, 2]])
    assert chosen["threshold"] == .7
    assert chosen["status"] == "review_target_unmet" and not chosen["objective_met"]


def test_selection_reports_infeasible_accuracy_target():
    chosen = select_setting(selection_rows().iloc[[0]])
    assert chosen["status"] == "accuracy_target_unmet" and not chosen["objective_met"]


def test_selection_never_treats_accepting_nothing_as_perfect():
    rows = selection_rows().iloc[[0]].copy()
    rows[["accepted_count", "correct_accepted", "coverage"]] = 0
    rows["review_rate"] = 1
    rows["accuracy_on_accepted"] = None
    with pytest.raises(ValueError, match="No setting accepts"):
        select_setting(rows)


def test_joint_search_scores_both_gate_parameters():
    values = GatePredictions(np.array(["Coffee", "Dining"]), np.array(["Coffee", "Dining"]),
                             np.array([.8, .6]), np.array([.2, .7]), np.array([UNCERTAIN_LABEL] * 2))
    sweep = joint_sweep(values, thresholds=[.5, .7], oov_cutoffs=[.6, .8])
    assert len(sweep) == 4
    assert sweep["accepted_count"].tolist() == [1, 2, 1, 1]


def test_reliability_includes_zero_and_one_and_weights_bins_by_count():
    table, ece = reliability_table([0.0, .2, .99, 1.0], [False, True, False, True], bins=5)
    assert table["count"].tolist() == [1, 1, 0, 0, 2]
    assert table.iloc[-1]["mean_confidence"] == pytest.approx(.995)
    assert table.iloc[-1]["accuracy"] == .5
    assert ece == pytest.approx(.4475)
    assert np.isnan(table.iloc[2]["accuracy"])


def test_reliability_rejects_invalid_probabilities_and_shapes():
    for confidence, correct in [([], []), ([.5], [True, False]), ([np.nan], [True]), ([1.1], [True]), ([.5], [2])]:
        with pytest.raises(ValueError):
            reliability_table(confidence, correct)


def test_settings_and_selection_survive_save_and_legacy_models_load(tmp_path):
    model = TransactionCategorizer(.72, .45)
    model.gate_selection_ = {"source": "validation"}
    model.vocabulary_ = {"coffee"}
    path = tmp_path / "model.joblib"
    model.save(path)
    restored = TransactionCategorizer.load(path)
    assert (restored.threshold, restored.oov_cutoff) == (.72, .45)
    assert restored.gate_selection_ == model.gate_selection_
    assert restored.vocabulary_ == model.vocabulary_
    joblib.dump({"pipeline": None, "threshold": .55, "vocabulary": set()}, path)
    legacy = TransactionCategorizer.load(path)
    assert legacy.oov_cutoff == .6 and legacy.gate_selection_ == {}


def test_explicit_model_flows_through_batch_and_budget_rollup():
    from src.categorize import categorize_batch, spending_by_budget_category
    transactions = [{"description": "custom description", "direction": "debit", "amount": 12.5}]
    models = []
    for category in ["Coffee", "Shopping"]:
        model = TransactionCategorizer()
        model._probabilities = lambda descriptions, label=category: (np.ones((len(descriptions), 1)), np.array([label]))
        model.oov_ratio = lambda description: 0.0
        models.append(model)
    assert categorize_batch(transactions, model=models[0])[0].source_category == "Coffee"
    assert categorize_batch(transactions, model=models[1])[0].source_category == "Shopping"
    assert spending_by_budget_category(transactions, model=models[0]) == {"Food & Drink": 12.5}
    assert spending_by_budget_category(transactions, model=models[1]) == {"Shopping": 12.5}


def test_evaluation_matches_public_api_for_saved_candidate():
    from src.categorize import categorize_batch
    from src.ml_model import DEFAULT_MODEL_PATH
    path = DEFAULT_MODEL_PATH.with_name("categorizer_candidate.joblib")
    if not path.exists():
        pytest.skip("run scripts/tune_gate.py to build the candidate")
    root = path.parent.parent
    frame = pd.concat([pd.read_csv(root / "data/test.csv").head(15),
                       pd.read_csv(root / "data/unseen_merchants.csv").head(15)], ignore_index=True)
    model = TransactionCategorizer.load(path)
    values = GatePredictions.from_frame(model, frame)
    accepted = acceptance_mask(values.confidence, values.oov, model.threshold, model.oov_cutoff)
    expected = np.where(accepted, values.predicted, values.fallback).tolist()
    results = categorize_batch(frame.to_dict("records"), model=model)
    assert [result.source_category for result in results] == expected
    counts = values.score(model.threshold, model.oov_cutoff, use_rules=True)["decided_by"]
    assert sum(result.needs_review for result in results) == counts["review"]
