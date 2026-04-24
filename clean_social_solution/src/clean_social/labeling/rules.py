from __future__ import annotations

import re

VALID_LABELS = {"Positive", "Neutral", "Negative"}
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "cant",
    "can't",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "won't",
    "wont",
    "n't",
}

_NEGATION_TOKEN_REGEX = re.compile(r"[a-zA-Z']+")


def normalize_label(value: str) -> str:
    label = str(value).strip().capitalize()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label '{value}'. Allowed labels: {sorted(VALID_LABELS)}")
    return label


def rating_to_sentiment(score: float) -> str:
    value = float(score)
    if value <= 2:
        return "Negative"
    if value == 3:
        return "Neutral"
    return "Positive"


def contains_negation(text: str) -> bool:
    tokens = _NEGATION_TOKEN_REGEX.findall(str(text).lower())
    return any(token in NEGATION_WORDS for token in tokens)
