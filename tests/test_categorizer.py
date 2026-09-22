"""
Tests.

Run:  python -m pytest tests/ -v
      (or: python tests/test_categorizer.py  to run without pytest)

These cover the behaviours that would silently break the budgeting app:
the taxonomy mapping, the direction rules, noise normalisation, and the
guarantee that an uncertain prediction never lands in a real budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules  # noqa: E402
from src.categories import (  # noqa: E402
    BUDGET_CATEGORIES,
    SOURCE_CATEGORIES,
    UNCERTAIN_LABEL,
    to_budget_category,
    validate_taxonomy,
)
from src.ml_model import DEFAULT_MODEL_PATH, normalize  # noqa: E402

MODEL_EXISTS = DEFAULT_MODEL_PATH.exists()
needs_model = pytest.mark.skipif(not MODEL_EXISTS, reason="run scripts/train.py first")


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
def test_taxonomy_is_consistent():
    validate_taxonomy()


def test_every_source_category_maps_to_a_budget_category():
    for category in SOURCE_CATEGORIES:
        assert to_budget_category(category) in BUDGET_CATEGORIES


def test_uncertain_label_never_becomes_a_real_budget():
    assert to_budget_category(UNCERTAIN_LABEL) == UNCERTAIN_LABEL


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        to_budget_category("Crypto Yacht Fund")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def test_normalize_strips_reference_codes_but_keeps_merchant():
    cleaned = normalize("[debit] AMZN MKTP US*2K4L91")
    assert "AMZN" in cleaned
    assert "2K4L91" not in cleaned


def test_normalize_keeps_direction_marker():
    assert "CREDIT" in normalize("[credit] VENMO CASHOUT*8823QK")


def test_normalize_strips_store_numbers():
    assert "4412" not in normalize("[debit] STARBUCKS #4412 WEST LAFAYETTE IN")


def test_normalize_collapses_whitespace():
    assert "  " not in normalize("[debit] SHELL   OIL    #7741")


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def test_rule_specificity_order():
    # The specific AWS rule must win over the general AMAZON rule.
    assert rules.classify("[debit] AMAZON WEB SERVICES DES:BILLPAY") == "Subscriptions"
    assert rules.classify("[debit] AMZN MKTP US*2K4L91") == "Shopping"


def test_rules_use_direction_for_peer_to_peer():
    assert rules.classify("[credit] VENMO CASHOUT*8823QK", "credit") == "Income"
    assert rules.classify("[debit] VENMO CASHOUT*8823QK", "debit") == "Transfer"


def test_rules_admit_ignorance_instead_of_guessing():
    assert rules.classify("[debit] SQ *MORNING GLORY BAKEHOUSE") == UNCERTAIN_LABEL


def test_uber_eats_is_dining_not_rideshare():
    assert rules.classify("[debit] UBER EATS*K92LM4") == "Dining"
    assert rules.classify("[debit] UBER TRIP*8M2QK4") == "Rideshare"


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------
@needs_model
def test_known_merchants_are_categorized_confidently():
    from src.categorize import categorize

    result = categorize("[debit] SQ *BLUE BOTTLE SAN FRANCISCO CA", 6.25)
    assert result.budget_category == "Food & Drink"
    assert result.confidence > 0.7
    assert not result.needs_review


@needs_model
def test_direction_flips_the_category():
    from src.categorize import categorize

    outgoing = categorize("[debit] VENMO CASHOUT*8823QK", 45.0, "debit")
    incoming = categorize("[credit] VENMO CASHOUT*4419MZ", 60.0, "credit")
    assert outgoing.source_category == "Transfer"
    assert incoming.source_category == "Income"


@needs_model
def test_unknown_merchant_is_flagged_not_guessed():
    from src.categorize import categorize

    result = categorize("[debit] SQ *ZZQX VENTURES LLC TPTF9S", 12.0)
    assert result.needs_review
    assert result.budget_category == UNCERTAIN_LABEL
    assert result.review_reason in {"unknown_merchant", "low_confidence"}


@needs_model
def test_budget_rollup_excludes_income():
    from src.categorize import spending_by_budget_category

    totals = spending_by_budget_category(
        [
            {"description": "[debit] CHIPOTLE #1882", "amount": 14.80, "direction": "debit"},
            {"description": "[credit] DIRECT DEP PAYROLL DES:DIRECT DEP", "amount": 1840.55,
             "direction": "credit"},
        ]
    )
    assert "Income" not in totals
    assert totals.get("Food & Drink") == 14.80


@needs_model
def test_every_result_is_a_valid_budget_category():
    from src.categorize import categorize_batch

    samples = [
        {"description": "[debit] KROGER #882 INDIANAPOLIS IN", "amount": 54.2, "direction": "debit"},
        {"description": "[debit] NETFLIX.COM*8821LM", "amount": 15.49, "direction": "debit"},
        {"description": "[debit] QQQ UNKNOWN VENDOR 99", "amount": 3.0, "direction": "debit"},
    ]
    for result in categorize_batch(samples):
        assert result.budget_category in BUDGET_CATEGORIES + [UNCERTAIN_LABEL]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
