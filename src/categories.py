"""
Category taxonomies for the budgeting app.

Two levels exist on purpose:

1. SOURCE_CATEGORIES  - the fine-grained labels the classifier is trained on.
                        These mirror the kind of taxonomy an aggregator such as
                        Plaid returns (PRIMARY / DETAILED personal finance
                        categories).
2. BUDGET_CATEGORIES  - the coarse buckets our app actually budgets against.
                        Users set a monthly limit per budget category, and the
                        alerting module fires against these.

Keeping them separate matters: when Plaid is integrated we can swap the
classifier out for Plaid's category and keep the mapping layer unchanged,
or fall back to the classifier whenever Plaid reports LOW confidence.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Level 1: fine-grained source categories (what the model predicts)
# --------------------------------------------------------------------------
SOURCE_CATEGORIES: list[str] = [
    "Groceries",
    "Dining",
    "Coffee",
    "Gas",
    "Rideshare",
    "Travel",
    "Shopping",
    "Subscriptions",
    "Entertainment",
    "Utilities",
    "Rent",
    "Healthcare",
    "Fitness",
    "Education",
    "Income",
    "Transfer",
    "Fees",
]

# --------------------------------------------------------------------------
# Level 2: app-facing budget categories (what the user sets limits on)
# --------------------------------------------------------------------------
BUDGET_CATEGORIES: list[str] = [
    "Food & Drink",
    "Transportation",
    "Bills & Housing",
    "Shopping",
    "Entertainment",
    "Health & Wellness",
    "Income",
    "Transfers & Fees",
]

# Explicit, auditable mapping. Every source category must appear here.
SOURCE_TO_BUDGET: dict[str, str] = {
    "Groceries": "Food & Drink",
    "Dining": "Food & Drink",
    "Coffee": "Food & Drink",
    "Gas": "Transportation",
    "Rideshare": "Transportation",
    "Travel": "Transportation",
    "Shopping": "Shopping",
    "Subscriptions": "Entertainment",
    "Entertainment": "Entertainment",
    "Utilities": "Bills & Housing",
    "Rent": "Bills & Housing",
    "Healthcare": "Health & Wellness",
    "Fitness": "Health & Wellness",
    "Education": "Bills & Housing",
    "Income": "Income",
    "Transfer": "Transfers & Fees",
    "Fees": "Transfers & Fees",
}

# The label used when the model is not confident enough to commit.
UNCERTAIN_LABEL = "Needs Review"


def to_budget_category(source_category: str) -> str:
    """Map a fine-grained source category onto an app budget category."""
    if source_category == UNCERTAIN_LABEL:
        return UNCERTAIN_LABEL
    try:
        return SOURCE_TO_BUDGET[source_category]
    except KeyError as exc:  # pragma: no cover - guards taxonomy drift
        raise ValueError(
            f"'{source_category}' is not in SOURCE_TO_BUDGET. "
            "Update src/categories.py when adding a category."
        ) from exc


def validate_taxonomy() -> None:
    """Fail loudly if the two taxonomies have drifted apart."""
    missing = set(SOURCE_CATEGORIES) - set(SOURCE_TO_BUDGET)
    if missing:
        raise ValueError(f"Source categories missing a budget mapping: {sorted(missing)}")

    unknown_targets = set(SOURCE_TO_BUDGET.values()) - set(BUDGET_CATEGORIES)
    if unknown_targets:
        raise ValueError(f"Mapping points at unknown budget categories: {sorted(unknown_targets)}")

    extra = set(SOURCE_TO_BUDGET) - set(SOURCE_CATEGORIES)
    if extra:
        raise ValueError(f"Mapping contains categories not in SOURCE_CATEGORIES: {sorted(extra)}")


validate_taxonomy()
