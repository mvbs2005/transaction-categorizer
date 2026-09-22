"""
Interactive demo - this is the script to record for the video.

    python scripts/demo.py                 # categorize a sample statement
    python scripts/demo.py --interactive   # type your own descriptions
    python scripts/demo.py --budgets       # show budget usage + alerts

Every transaction below is written in a real bank statement format, including
ones the model has never seen, so the review queue is visible in the output
rather than hidden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.categorize import categorize, categorize_batch, spending_by_budget_category  # noqa: E402

SAMPLE_STATEMENT = [
    {"description": "[debit] SQ *BLUE BOTTLE SAN FRANCISCO CA", "amount": 6.25, "direction": "debit"},
    {"description": "[debit] CHIPOTLE #1882 WEST LAFAYETTE IN", "amount": 14.80, "direction": "debit"},
    {"description": "[debit] AMZN MKTP US*2K4L91", "amount": 63.19, "direction": "debit"},
    {"description": "[debit] ADOBE INC DES:BILLPAY ID:7781QK2LMN INDN:CARDHOLDER", "amount": 22.99, "direction": "debit"},
    {"description": "[debit] SHELL OIL #7741 INDIANAPOLIS IN", "amount": 41.02, "direction": "debit"},
    {"description": "[debit] DUKE ENERGY DES:ACH PMT ID:99KL2MQ8ZX INDN:CARDHOLDER", "amount": 118.44, "direction": "debit"},
    {"description": "[debit] GREYSTAR PROPERTY DES:PAYMENT ID:41MQ8XZ2LK INDN:CARDHOLDER", "amount": 1180.00, "direction": "debit"},
    {"description": "[debit] PLANET FITNESS #204 WEST LAFAYETTE IN", "amount": 24.99, "direction": "debit"},
    {"description": "[debit] UBER TRIP*8M2QK4", "amount": 18.35, "direction": "debit"},
    {"description": "[debit] POS DEBIT CVS PHARMACY TPTF9S", "amount": 31.70, "direction": "debit"},
    {"description": "[credit] DIRECT DEP PAYROLL DES:DIRECT DEP ID:22KQ8M4LZX INDN:CARDHOLDER", "amount": 1840.55, "direction": "credit"},
    {"description": "[debit] VENMO CASHOUT*8823QK", "amount": 45.00, "direction": "debit"},
    {"description": "[credit] VENMO CASHOUT*4419MZ", "amount": 60.00, "direction": "credit"},
    # Merchants the model has never seen - these should surface as review items
    {"description": "[debit] STARBUCKS #4412 WEST LAFAYETTE IN", "amount": 5.45, "direction": "debit"},
    {"description": "[debit] WEGMANS #218 BROOKLYN NY", "amount": 87.31, "direction": "debit"},
    {"description": "[debit] SQ *MORNING GLORY BAKEHOUSE LAFAYETTE IN", "amount": 9.10, "direction": "debit"},
]

MONTHLY_BUDGETS = {
    "Food & Drink": 400.00,
    "Transportation": 150.00,
    "Bills & Housing": 1400.00,
    "Shopping": 200.00,
    "Entertainment": 60.00,
    "Health & Wellness": 80.00,
}


def run_statement() -> None:
    results = categorize_batch(SAMPLE_STATEMENT)

    print("\n" + "=" * 104)
    print(f"{'DESCRIPTION':<58}{'CATEGORY':<16}{'BUDGET':<19}{'CONF':>5}  SOURCE")
    print("=" * 104)

    for transaction, result in zip(SAMPLE_STATEMENT, results):
        description = transaction["description"]
        if len(description) > 56:
            description = description[:53] + "..."
        marker = f" <-- {result.review_reason}" if result.needs_review else ""
        print(
            f"{description:<58}{result.source_category:<16}{result.budget_category:<19}"
            f"{result.confidence:>5.2f}  {result.decided_by}{marker}"
        )

    review_count = sum(1 for r in results if r.needs_review)
    by_rules = sum(1 for r in results if r.decided_by == "rules")
    print("-" * 104)
    print(
        f"{len(results)} transactions | "
        f"{len(results) - review_count - by_rules} decided by the model | "
        f"{by_rules} by rule fallback | {review_count} sent to review"
    )

    print("\nWhy the review items matter: each one is a merchant the model has")
    print("never seen. The system declines to guess rather than silently")
    print("dropping the wrong amount into a budget category.\n")


def run_budgets() -> None:
    totals = spending_by_budget_category(SAMPLE_STATEMENT)

    print("\n" + "=" * 62)
    print(f"{'BUDGET CATEGORY':<22}{'SPENT':>11}{'LIMIT':>11}{'USED':>9}  STATUS")
    print("=" * 62)

    for category, limit in MONTHLY_BUDGETS.items():
        spent = totals.get(category, 0.0)
        pct = spent / limit if limit else 0
        if pct >= 1.0:
            status = "OVER BUDGET"
        elif pct >= 0.8:
            status = "approaching limit"
        else:
            status = "ok"
        print(f"{category:<22}{spent:>11,.2f}{limit:>11,.2f}{pct:>8.0%}  {status}")

    unresolved = totals.get("Needs Review", 0.0)
    if unresolved:
        print("-" * 62)
        print(f"{'Needs Review':<22}{unresolved:>11,.2f}{'':>11}{'':>9}  awaiting user confirmation")

    print("\nThis rollup is the hand-off point to the alerting module: it takes")
    print("per-category spend and decides when to notify the user.\n")


def run_interactive() -> None:
    print("\nType a transaction description (blank line to quit).")
    print("Try:  AMZN MKTP US*2K4L91   |   SQ *BLUE BOTTLE SF CA   |   DUKE ENERGY DES:ACH PMT\n")
    while True:
        try:
            raw = input("description> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            return

        if not raw.lower().startswith("["):
            raw = f"[debit] {raw}"

        result = categorize(raw)
        print(f"  category   : {result.source_category}")
        print(f"  budget     : {result.budget_category}")
        print(f"  confidence : {result.confidence:.3f}")
        print(f"  decided by : {result.decided_by}"
              + (f" ({result.review_reason})" if result.review_reason else ""))
        alternatives = ", ".join(f"{c} {p:.2f}" for c, p in result.alternatives)
        print(f"  top 3      : {alternatives}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--budgets", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.budgets:
        run_statement()
        run_budgets()
    else:
        run_statement()


if __name__ == "__main__":
    main()
