"""
Rule-based baseline.

This is the approach most budgeting apps start with: a hand-written keyword
table. It is included so the machine-learning model has an honest baseline to
beat, and so the report can quantify *how much* the model is worth.

The rule list below is intentionally realistic rather than exhaustive: it is
about the size of a table one developer would maintain by hand. Its failure
mode is the point of the experiment - it does well on merchants someone
remembered to add and falls back to a default everywhere else.
"""

from __future__ import annotations

import re

from .categories import UNCERTAIN_LABEL

# Ordered rules: the first matching pattern wins, so put the specific
# exceptions (AMAZON WEB SERVICES) above the general ones (AMAZON).
KEYWORD_RULES: list[tuple[str, str]] = [
    # --- specific overrides first -----------------------------------------
    (r"AMAZON WEB SERVICES|AWS", "Subscriptions"),
    (r"AMAZON|AMZN", "Shopping"),
    (r"UBER EATS|UBEREATS", "Dining"),
    (r"UBER TRIP|UBER \*|LYFT", "Rideshare"),
    # --- food -------------------------------------------------------------
    (r"KROGER|TRADER JOES|WHOLE FOODS|ALDI|PUBLIX|SAFEWAY|MEIJER|WEGMANS", "Groceries"),
    (r"CHIPOTLE|PANDA EXPRESS|OLIVE GARDEN|DOORDASH|GRUBHUB|WINGSTOP|FIVE GUYS", "Dining"),
    (r"STARBUCKS|DUNKIN|BLUE BOTTLE|PEETS|CARIBOU|TIM HORTONS", "Coffee"),
    # --- transport --------------------------------------------------------
    (r"SHELL OIL|SPEEDWAY|\bBP\b|EXXON|CIRCLE K|MARATHON PETRO|SUNOCO", "Gas"),
    (r"DELTA AIR|UNITED AIRLINES|SOUTHWEST|MARRIOTT|HILTON|AIRBNB|EXPEDIA|AMTRAK", "Travel"),
    (r"PARKING|VENTRA|PARKMOBILE|SPOTHERO", "Rideshare"),
    # --- bills ------------------------------------------------------------
    (r"DUKE ENERGY|XFINITY|VERIZON|AT&T|NIPSCO|SPECTRUM|T-MOBILE|WATER WORKS", "Utilities"),
    (r"GREYSTAR|APTS|PROPERTY|RESIDENTIAL|CAMPUS CROSSING", "Rent"),
    # --- discretionary ----------------------------------------------------
    (r"NETFLIX|SPOTIFY|ADOBE|HULU|DISNEY PLUS|GITHUB|DROPBOX|NYTIMES", "Subscriptions"),
    (r"AMC |STEAMGAMES|TICKETMASTER|REGAL CINEMAS|LIVE NATION|PLAYSTATION", "Entertainment"),
    (r"BEST BUY|HOME DEPOT|IKEA|NIKE|ETSY|MACYS|WAYFAIR|LOWES|TARGET", "Shopping"),
    # --- health / education ----------------------------------------------
    (r"CVS|WALGREENS|QUEST DIAGNOSTICS|LABCORP|ASPEN DENTAL|PHARMACY|URGENT CARE", "Healthcare"),
    (r"PLANET FITNESS|ANYTIME FITNESS|ORANGETHEORY|LA FITNESS|YMCA|CLASSPASS", "Fitness"),
    (r"UNIVERSITY|COURSERA|UDEMY|CHEGG|PEARSON|MCGRAW HILL|BOOKSTORE", "Education"),
    # --- money movement ---------------------------------------------------
    (r"PAYROLL|DIRECT DEP|TREAS 310|PAYOUT|DIVIDEND", "Income"),
    (r"VENMO|ZELLE|CASH APP|WIRE TRANSFER OUT|ONLINE XFER|COINBASE|ROBINHOOD", "Transfer"),
    (r"\bFEE\b|OVERDRAFT|SURCHARGE", "Fees"),
]

COMPILED_RULES = [(re.compile(pattern), category) for pattern, category in KEYWORD_RULES]

# What to do when nothing matches. Two honest options: guess the most common
# category, or admit ignorance. We admit ignorance and count it as an error,
# which is how the alerting module would actually have to treat it.
DEFAULT_CATEGORY = UNCERTAIN_LABEL


def classify(description: str, direction: str | None = None) -> str:
    """Return a source category for a raw transaction description."""
    text = description.upper()

    # Direction overrides: a Venmo credit is income, not a transfer.
    if direction == "credit" and re.search(r"VENMO|ZELLE|CASH APP|PAYPAL INST|APPLE CASH", text):
        return "Income"

    for pattern, category in COMPILED_RULES:
        if pattern.search(text):
            return category
    return DEFAULT_CATEGORY


def classify_many(descriptions: list[str], directions: list[str] | None = None) -> list[str]:
    if directions is None:
        directions = [None] * len(descriptions)
    return [classify(d, dir_) for d, dir_ in zip(descriptions, directions)]


def rule_count() -> int:
    return len(KEYWORD_RULES)
