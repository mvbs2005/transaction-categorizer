"""
The machine-learning categorizer.

Design choices and why they were made:

* TF-IDF over word n-grams AND character n-grams.
  Word features capture merchant tokens ("KROGER", "NETFLIX"); character
  features survive the abbreviations and glued-together strings that word
  tokenisation destroys ("AMZN MKTP US*2K4L91"). Combining both is what makes
  a sparse linear model competitive with far heavier neural approaches on this
  kind of short, noisy text.

* Logistic regression rather than a transformer.
  The published work on this task finds sparse TF-IDF plus a calibrated linear
  model matches or beats sentence-embedding approaches while being orders of
  magnitude faster and small enough to ship inside the app.

* Probability calibration.
  A raw logistic-regression score is not a trustworthy probability. Calibrating
  it means "0.9 confidence" really does correspond to being right about 90% of
  the time, which is the whole basis of the confidence gate below.

* A confidence gate plus an out-of-vocabulary check.
  Silent misclassification is worse than admitting uncertainty: a wrong
  category quietly corrupts a budget, while an uncertain one can be sent back
  to the user for a one-tap confirmation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .categories import UNCERTAIN_LABEL, to_budget_category

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "categorizer.joblib"
DEFAULT_THRESHOLD = 0.55

# Tokens that carry no signal: store numbers, reference codes, dates.
_NOISE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), " "),      # dates
    (re.compile(r"#\d+"), " "),                       # store numbers
    (re.compile(r"\bID:[A-Z0-9]+"), " ACHID "),       # ACH identifiers
    # Reference codes: 5+ character tokens mixing letters AND digits.
    # The lookaheads are what keep real merchant names ("STARBUCKS") intact
    # while collapsing junk like "2K4L91" - an earlier version of this regex
    # matched any long uppercase token and deleted every merchant in the
    # dataset, which cost the model ~25 points of accuracy.
    (re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,}\b"), " REFCODE "),
    (re.compile(r"\d+"), " "),                        # any remaining digits
    (re.compile(r"[^A-Z\[\]\s]"), " "),               # punctuation, keep [debit]
    (re.compile(r"\s+"), " "),
]


def normalize(description: str) -> str:
    """Strip per-transaction noise so the merchant signal survives.

    Regex normalisation before vectorising is what stops the model from
    memorising unique reference codes, and it prevents near-duplicate strings
    from leaking between the train and test splits.
    """
    text = description.upper()
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


@dataclass
class Prediction:
    source_category: str
    budget_category: str
    confidence: float
    uncertain: bool
    top_alternatives: list[tuple[str, float]]
    reason: str = ""  # "", "low_confidence" or "unknown_merchant"


def build_pipeline() -> Pipeline:
    """Vectoriser + calibrated linear classifier."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )

    base = LogisticRegression(
        C=8.0,
        max_iter=2000,
        class_weight="balanced",
    )

    classifier = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    return Pipeline([("features", features), ("clf", classifier)])


class TransactionCategorizer:
    """Train, persist and query the categorizer."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.pipeline: Pipeline | None = None
        self.threshold = threshold
        self.vocabulary_: set[str] = set()

    # -- training ---------------------------------------------------------
    def fit(self, descriptions: list[str], labels: list[str]) -> "TransactionCategorizer":
        texts = [normalize(d) for d in descriptions]
        self.pipeline = build_pipeline()
        self.pipeline.fit(texts, labels)

        word_vec = self.pipeline.named_steps["features"].transformer_list[0][1]
        self.vocabulary_ = set(word_vec.vocabulary_.keys())
        return self

    # -- inference --------------------------------------------------------
    def _probabilities(self, descriptions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() or load().")
        texts = [normalize(d) for d in descriptions]
        probabilities = self.pipeline.predict_proba(texts)
        classes = self.pipeline.named_steps["clf"].classes_
        return probabilities, classes

    def predict(self, descriptions: list[str]) -> list[str]:
        """Best-guess source category, ignoring the confidence gate."""
        probabilities, classes = self._probabilities(descriptions)
        return [classes[i] for i in probabilities.argmax(axis=1)]

    def predict_with_confidence(self, descriptions: list[str]) -> list[Prediction]:
        probabilities, classes = self._probabilities(descriptions)
        results: list[Prediction] = []

        for row, description in zip(probabilities, descriptions):
            order = np.argsort(row)[::-1]
            best_index = order[0]
            confidence = float(row[best_index])
            category = str(classes[best_index])

            low_coverage = self.oov_ratio(description) > 0.6
            uncertain = confidence < self.threshold or low_coverage

            if not uncertain:
                reason = ""
            elif low_coverage:
                # An unknown merchant is flagged even when the character
                # features happen to produce a confident-looking score.
                reason = "unknown_merchant"
            else:
                reason = "low_confidence"

            results.append(
                Prediction(
                    source_category=UNCERTAIN_LABEL if uncertain else category,
                    budget_category=(
                        UNCERTAIN_LABEL if uncertain else to_budget_category(category)
                    ),
                    confidence=round(confidence, 4),
                    uncertain=uncertain,
                    top_alternatives=[
                        (str(classes[i]), round(float(row[i]), 4)) for i in order[:3]
                    ],
                    reason=reason,
                )
            )
        return results

    def oov_ratio(self, description: str) -> float:
        """Fraction of word tokens the model has never seen.

        A description made entirely of unknown tokens is a new merchant. Even
        if the character features produce a confident guess, that guess
        deserves a second look before it silently lands in someone's budget.
        """
        if not self.vocabulary_:
            return 0.0
        tokens = [t for t in normalize(description).split() if len(t) > 2]
        if not tokens:
            return 1.0
        unknown = sum(1 for t in tokens if t.lower() not in self.vocabulary_)
        return unknown / len(tokens)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": self.threshold,
                "vocabulary": self.vocabulary_,
            },
            path,
            compress=3,
        )
        meta = {
            "threshold": self.threshold,
            "vocabulary_size": len(self.vocabulary_),
            "size_kb": round(path.stat().st_size / 1024, 1),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "TransactionCategorizer":
        payload = joblib.load(Path(path))
        model = cls(threshold=payload["threshold"])
        model.pipeline = payload["pipeline"]
        model.vocabulary_ = payload["vocabulary"]
        return model
