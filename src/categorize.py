"""
Public API for the rest of the project.

This is the single function the other modules import. Everything else in this
package is an implementation detail.

    from src.categorize import categorize

    result = categorize("[debit] SQ *BLUE BOTTLE SAN FRANCISCO CA", amount=6.25)
    result.budget_category   -> "Food & Drink"
    result.confidence        -> 0.97
    result.needs_review      -> False

The fallback chain is ordered by cost:

    1. trained classifier  (instant, free, most accurate here)
    2. keyword rules       (only consulted when the model is unsure)
    3. needs_review        (handed back to the user for one-tap confirmation)

When Plaid is integrated, its category becomes step 0 and this module keeps
working unchanged: anything Plaid returns with LOW confidence simply drops
through to step 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import rules
from .categories import UNCERTAIN_LABEL, to_budget_category
from .ml_model import DEFAULT_MODEL_PATH, TransactionCategorizer

_model: TransactionCategorizer | None = None


@dataclass
class CategoryResult:
    description: str
    source_category: str
    budget_category: str
    confidence: float
    needs_review: bool
    decided_by: str
    review_reason: str = ""
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "source_category": self.source_category,
            "budget_category": self.budget_category,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "decided_by": self.decided_by,
            "review_reason": self.review_reason,
            "alternatives": self.alternatives,
        }


def load_model(path: Path = DEFAULT_MODEL_PATH) -> TransactionCategorizer:
    """Load the trained model once and cache it for the process lifetime."""
    global _model
    if _model is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python scripts/train.py"
            )
        _model = TransactionCategorizer.load(path)
    return _model


def categorize(
    description: str,
    amount: float = 0.0,
    direction: str | None = None,
    use_rule_fallback: bool = True,
) -> CategoryResult:
    """Categorize a single transaction description."""
    model = load_model()

    if direction is None:
        direction = "credit" if "[credit]" in description.lower() else "debit"

    prediction = model.predict_with_confidence([description])[0]

    if not prediction.uncertain:
        return CategoryResult(
            description=description,
            source_category=prediction.source_category,
            budget_category=prediction.budget_category,
            confidence=prediction.confidence,
            needs_review=False,
            decided_by="model",
            alternatives=prediction.top_alternatives,
        )

    if use_rule_fallback:
        rule_category = rules.classify(description, direction)
        if rule_category != UNCERTAIN_LABEL:
            return CategoryResult(
                description=description,
                source_category=rule_category,
                budget_category=to_budget_category(rule_category),
                confidence=prediction.confidence,
                needs_review=False,
                decided_by="rules",
                alternatives=prediction.top_alternatives,
            )

    return CategoryResult(
        description=description,
        source_category=UNCERTAIN_LABEL,
        budget_category=UNCERTAIN_LABEL,
        confidence=prediction.confidence,
        needs_review=True,
        decided_by="needs_review",
        review_reason=prediction.reason,
        alternatives=prediction.top_alternatives,
    )


def categorize_batch(transactions: list[dict]) -> list[CategoryResult]:
    """Categorize many transactions.

    Each item needs a "description" key; "amount" and "direction" are optional.
    This is the entry point the transaction-fetching module will call once
    Plaid returns a page of transactions.
    """
    return [
        categorize(
            t["description"],
            amount=t.get("amount", 0.0),
            direction=t.get("direction"),
        )
        for t in transactions
    ]


def spending_by_budget_category(transactions: list[dict]) -> dict[str, float]:
    """Roll categorized transactions up into per-budget-category totals.

    This is the shape the budget-alerting module consumes.
    """
    totals: dict[str, float] = {}
    for transaction, result in zip(transactions, categorize_batch(transactions)):
        if transaction.get("direction") == "credit":
            continue  # money in is not spending
        totals[result.budget_category] = round(
            totals.get(result.budget_category, 0.0) + float(transaction.get("amount", 0.0)), 2
        )
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))
