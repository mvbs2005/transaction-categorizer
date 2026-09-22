"""
Synthetic transaction generator.

Real bank transaction data is private, but the *formats* are public and
predictable: Chase, Capital One, Apple Card, PayPal and ACH all emit
descriptions with recognisable structures. This module reproduces those
structures over a merchant catalogue so we can train and evaluate a
classifier before Plaid is wired into the project.

Three properties are deliberate, because they are what make the task
non-trivial and the evaluation honest:

1. NOISE       - store numbers, reference codes, city/state suffixes and
                 payment-processor wrappers are randomised, so the model
                 cannot succeed by memorising whole strings.
2. AMBIGUITY   - some merchants genuinely belong to more than one category
                 (TARGET is groceries or general shopping; AMAZON is shopping
                 but AMAZON WEB SERVICES is a subscription). This puts a
                 realistic ceiling below 100% accuracy.
3. DIRECTION   - "VENMO CASHOUT" is income as a credit and a transfer as a
                 debit. The debit/credit prefix is part of the input so the
                 model has to learn to use it.

A fixed fraction of merchants in every category is reserved as
HELD-OUT MERCHANTS. Those never appear in training data; they form the
unseen-merchant slice used to measure real generalisation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

from .categories import SOURCE_TO_BUDGET

# --------------------------------------------------------------------------
# Merchant catalogue
# --------------------------------------------------------------------------
MERCHANTS: dict[str, list[str]] = {
    "Groceries": [
        "KROGER", "TRADER JOES", "WHOLE FOODS MKT", "ALDI", "PUBLIX",
        "SAFEWAY", "MEIJER", "HEB ONLINE", "WEGMANS", "SPROUTS FARMERS MKT",
        "FOOD LION", "GIANT EAGLE", "WINCO FOODS", "STOP N SHOP", "HARRIS TEETER",
    ],
    "Dining": [
        "CHIPOTLE", "PANDA EXPRESS", "OLIVE GARDEN", "SHAKE SHACK", "TST* BAR LOUIE",
        "SQ *TACO CASA", "DOORDASH", "UBER EATS", "GRUBHUB", "WINGSTOP",
        "FIVE GUYS", "NOODLES AND CO", "TST* THE BLUE DOOR", "SWEETGREEN", "RAISING CANES",
    ],
    "Coffee": [
        "STARBUCKS", "DUNKIN", "SQ *BLUE BOTTLE", "PEETS COFFEE", "TST* GREYHOUSE",
        "CARIBOU COFFEE", "SQ *LUMINARY BAKERY", "TIM HORTONS", "SCOOTERS COFFEE",
        "SQ *VIENNA ESPRESSO", "BIGGBY COFFEE",
    ],
    "Gas": [
        "SHELL OIL", "SPEEDWAY", "BP PRODUCTS", "EXXONMOBIL", "CIRCLE K",
        "MARATHON PETRO", "CASEYS GEN STORE", "PILOT TRAVEL CTR", "SUNOCO",
        "QUIKTRIP", "WAWA",
    ],
    "Rideshare": [
        "UBER TRIP", "LYFT RIDE", "CITY PARKING AUTH", "CTA VENTRA", "BIRD RIDES",
        "LIME RIDE", "CURB TAXI", "MTA VENDING MACH", "PARKMOBILE", "SPOTHERO",
    ],
    "Travel": [
        "DELTA AIR LINES", "UNITED AIRLINES", "SOUTHWEST AIRLINE", "MARRIOTT HOTELS",
        "HILTON GARDEN INN", "AIRBNB", "EXPEDIA", "HERTZ RENT A CAR", "AMTRAK",
        "BOOKING.COM", "ALASKA AIR", "ENTERPRISE RENT A CAR",
    ],
    "Shopping": [
        "AMAZON MKTPL", "AMZN MKTP US", "BEST BUY", "HOME DEPOT", "IKEA",
        "NIKE.COM", "ETSY.COM", "MACYS", "WAYFAIR", "LOWES",
        "REI.COM", "UNIQLO USA", "DICKS SPORTING GDS", "BARNES NOBLE",
    ],
    "Subscriptions": [
        "NETFLIX.COM", "SPOTIFY USA", "ADOBE INC", "AMAZON WEB SERVICES", "GITHUB INC",
        "OPENAI CHATGPT SUB", "DROPBOX", "NYTIMES DIGITAL", "HULU", "NOTION LABS",
        "DISNEY PLUS", "ICLOUD STORAGE",
    ],
    "Entertainment": [
        "AMC ONLINE", "STEAMGAMES.COM", "TICKETMASTER", "REGAL CINEMAS", "DAVE AND BUSTERS",
        "LIVE NATION", "PLAYSTATION NETWORK", "TOPGOLF", "MAIN EVENT", "EVENTBRITE",
        "XBOX GAME PASS",
    ],
    "Utilities": [
        "DUKE ENERGY", "COMCAST XFINITY", "VERIZON WIRELESS", "AT&T MOBILITY", "NIPSCO",
        "CITY WATER WORKS", "SPECTRUM", "T-MOBILE", "AMERICAN ELECTRIC", "REPUBLIC SERVICES",
        "GOOGLE FIBER",
    ],
    "Rent": [
        "GREYSTAR PROPERTY", "CAMPUS CROSSING", "AVALON COMMUNITIES", "LINCOLN PROP CO",
        "WILLOW CREEK APTS", "PROGRESS RESIDENTIAL", "THE LODGE APTS", "CORTLAND MGMT",
        "REDSTONE PROPERTIES",
    ],
    "Healthcare": [
        "CVS PHARMACY", "WALGREENS", "QUEST DIAGNOSTICS", "IU HEALTH", "LABCORP",
        "ASPEN DENTAL", "ONE MEDICAL", "ZOCDOC", "OPTUM RX", "VISION WORKS",
        "URGENT CARE PARTNRS",
    ],
    "Fitness": [
        "PLANET FITNESS", "ANYTIME FITNESS", "ORANGETHEORY", "CRUNCH FITNESS",
        "PELOTON MEMBERSHIP", "CLASSPASS", "LA FITNESS", "YMCA MEMBERSHIP",
        "F45 TRAINING", "STRAVA SUBSCRIPTION",
    ],
    "Education": [
        "PURDUE UNIVERSITY", "COURSERA INC", "UDEMY", "PEARSON EDUCATION",
        "CHEGG ORDER", "BARNES NOBLE COLLEG", "MCGRAW HILL", "EDX INC",
        "DATACAMP", "UNIV BOOKSTORE",
    ],
    "Income": [
        "PAYROLL DEP", "DIRECT DEP PAYROLL", "IRS TREAS 310 TAX REF", "STRIPE TRANSFER",
        "UPWORK ESCROW", "STATE OF IN PAYROLL", "DOORDASH DRIVER PAY", "VA BENEFITS",
        "FIDELITY DIVIDEND", "ETSY PAYOUT",
    ],
    "Transfer": [
        "VENMO CASHOUT", "ZELLE PAYMENT", "CASH APP TRANSFER", "PAYPAL INST XFER",
        "WIRE TRANSFER OUT", "ONLINE XFER TO SAV", "APPLE CASH TRANSFER",
        "ROBINHOOD FUNDING", "COINBASE TRANSFER", "SOFI MONEY XFER",
    ],
    "Fees": [
        "MONTHLY MAINT FEE", "OVERDRAFT FEE", "ATM SURCHARGE", "FOREIGN TRANS FEE",
        "LATE PAYMENT FEE", "WIRE TRANSFER FEE", "CARD REPLACEMENT FEE",
        "RETURNED ITEM FEE", "PAPER STATEMENT FEE", "STOP PAYMENT FEE",
    ],
}

# Merchants that legitimately span two categories. The classifier cannot be
# perfect on these, which is exactly why a confidence score is needed.
AMBIGUOUS_MERCHANTS: list[tuple[str, dict[str, float]]] = [
    ("TARGET", {"Groceries": 0.45, "Shopping": 0.55}),
    ("WALMART SUPERCENTER", {"Groceries": 0.6, "Shopping": 0.4}),
    ("COSTCO WHSE", {"Groceries": 0.55, "Gas": 0.2, "Shopping": 0.25}),
    ("SAMS CLUB", {"Groceries": 0.5, "Shopping": 0.5}),
    ("APPLE.COM/BILL", {"Subscriptions": 0.65, "Shopping": 0.35}),
    ("WALGREENS", {"Healthcare": 0.6, "Shopping": 0.4}),
    ("SPEEDWAY", {"Gas": 0.8, "Groceries": 0.2}),
]

# Merchants whose category flips with the direction of the transaction.
DIRECTION_SENSITIVE: dict[str, tuple[str, str]] = {
    # merchant: (category when debit, category when credit)
    "VENMO CASHOUT": ("Transfer", "Income"),
    "ZELLE PAYMENT": ("Transfer", "Income"),
    "CASH APP TRANSFER": ("Transfer", "Income"),
    "PAYPAL INST XFER": ("Transfer", "Income"),
    "APPLE CASH TRANSFER": ("Transfer", "Income"),
}

CITIES: list[tuple[str, str]] = [
    ("WEST LAFAYETTE", "IN"), ("INDIANAPOLIS", "IN"), ("CHICAGO", "IL"),
    ("SAN FRANCISCO", "CA"), ("AUSTIN", "TX"), ("SEATTLE", "WA"),
    ("COLUMBUS", "OH"), ("ATLANTA", "GA"), ("DENVER", "CO"),
    ("BROOKLYN", "NY"), ("PHOENIX", "AZ"), ("NASHVILLE", "TN"),
]

AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "Groceries": (8.0, 210.0),
    "Dining": (7.0, 95.0),
    "Coffee": (2.5, 24.0),
    "Gas": (12.0, 88.0),
    "Rideshare": (3.5, 62.0),
    "Travel": (45.0, 1250.0),
    "Shopping": (6.0, 480.0),
    "Subscriptions": (2.99, 59.99),
    "Entertainment": (9.0, 180.0),
    "Utilities": (25.0, 310.0),
    "Rent": (650.0, 2400.0),
    "Healthcare": (10.0, 640.0),
    "Fitness": (10.0, 160.0),
    "Education": (18.0, 1800.0),
    "Income": (120.0, 4200.0),
    "Transfer": (20.0, 900.0),
    "Fees": (2.0, 39.0),
}

CREDIT_CATEGORIES = {"Income"}


@dataclass
class Transaction:
    description: str
    amount: float
    direction: str            # "debit" or "credit"
    source_category: str
    budget_category: str
    merchant: str
    merchant_seen_in_training: bool


# --------------------------------------------------------------------------
# Description formatting
# --------------------------------------------------------------------------
def _ref_code(rng: random.Random, length: int = 8) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _format_description(rng: random.Random, merchant: str, category: str) -> str:
    """Wrap a merchant name in one of several real bank statement formats."""
    city, state = rng.choice(CITIES)
    style = rng.random()

    # ACH / direct-deposit formats are shared by several recurring categories.
    # If only Rent/Utilities/Income used this format the model would learn the
    # format instead of the merchant, so recurring billing categories share it.
    ach_categories = {"Rent", "Utilities", "Income", "Subscriptions", "Fitness", "Education"}
    if category in ach_categories and style < 0.45:
        # ACH / direct deposit style
        purpose = rng.choice(["PAYMENT", "DIRECT DEP", "ACH PMT", "BILLPAY", "DEPOSIT"])
        return f"{merchant} DES:{purpose} ID:{_ref_code(rng, 10)} INDN:CARDHOLDER"

    if category == "Fees":
        return merchant if style < 0.7 else f"{merchant} {_ref_code(rng, 4)}"

    if style < 0.15:
        # PayPal wrapper
        return f"PAYPAL *{merchant.replace(' ', '')[:18]}"
    if style < 0.28:
        # Square wrapper (skip if already wrapped)
        if not merchant.startswith(("SQ *", "TST*")):
            return f"SQ *{merchant} {city[:10]} {state}"
    if style < 0.45:
        # Store number + city/state
        return f"{merchant} #{rng.randint(100, 9999)} {city} {state}"
    if style < 0.6:
        # Online order reference
        return f"{merchant}*{_ref_code(rng, rng.randint(6, 10))}"
    if style < 0.72:
        # Purchase prefix
        return f"PURCHASE {merchant} {city} {state}US"
    if style < 0.85:
        return f"{merchant} {city} {state}"
    return f"POS DEBIT {merchant} {_ref_code(rng, 6)}"


def _pick_category(rng: random.Random, merchant: str, base_category: str, direction: str) -> str:
    if merchant in DIRECTION_SENSITIVE:
        debit_cat, credit_cat = DIRECTION_SENSITIVE[merchant]
        return credit_cat if direction == "credit" else debit_cat
    return base_category


# --------------------------------------------------------------------------
# Catalogue split
# --------------------------------------------------------------------------
def split_merchant_catalogue(
    holdout_fraction: float = 0.25, seed: int = 13
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split every category's merchant list into seen and held-out merchants."""
    rng = random.Random(seed)
    seen: dict[str, list[str]] = {}
    held_out: dict[str, list[str]] = {}

    for category, merchants in MERCHANTS.items():
        shuffled = merchants[:]
        rng.shuffle(shuffled)
        n_holdout = max(1, int(round(len(shuffled) * holdout_fraction)))
        held_out[category] = sorted(shuffled[:n_holdout])
        seen[category] = sorted(shuffled[n_holdout:])

    return seen, held_out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate(
    catalogue: dict[str, list[str]],
    rows_per_category: int,
    seed: int,
    merchant_seen_in_training: bool,
    include_ambiguous: bool = True,
) -> list[Transaction]:
    """Generate transactions from a merchant catalogue."""
    rng = random.Random(seed)
    rows: list[Transaction] = []

    for base_category, merchants in catalogue.items():
        if not merchants:
            continue

        pool: list[str] = list(merchants)
        if include_ambiguous:
            for merchant, weights in AMBIGUOUS_MERCHANTS:
                if base_category in weights:
                    # Sample count proportional to that merchant's share.
                    pool.extend([merchant] * max(1, int(len(merchants) * weights[base_category])))

        for _ in range(rows_per_category):
            merchant = rng.choice(pool)

            if base_category in CREDIT_CATEGORIES:
                direction = "credit"
            elif merchant in DIRECTION_SENSITIVE:
                direction = "credit" if rng.random() < 0.4 else "debit"
            else:
                direction = "debit" if rng.random() < 0.97 else "credit"

            category = _pick_category(rng, merchant, base_category, direction)
            lo, hi = AMOUNT_RANGES[category]
            amount = round(rng.uniform(lo, hi), 2)
            body = _format_description(rng, merchant, category)
            description = f"[{direction}] {body}"

            rows.append(
                Transaction(
                    description=description,
                    amount=amount,
                    direction=direction,
                    source_category=category,
                    budget_category=SOURCE_TO_BUDGET[category],
                    merchant=merchant,
                    merchant_seen_in_training=merchant_seen_in_training,
                )
            )

    rng.shuffle(rows)
    return rows


def deduplicate(rows: list[Transaction]) -> list[Transaction]:
    """Drop exact duplicate descriptions.

    This matters more than it looks: duplicate strings that land on both
    sides of a train/test split leak the answer and inflate the score.
    """
    seen_descriptions: set[str] = set()
    unique: list[Transaction] = []
    for row in rows:
        if row.description in seen_descriptions:
            continue
        seen_descriptions.add(row.description)
        unique.append(row)
    return unique


def to_records(rows: list[Transaction]) -> list[dict]:
    return [asdict(row) for row in rows]
