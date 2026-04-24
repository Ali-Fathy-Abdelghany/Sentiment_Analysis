from __future__ import annotations

import re
from pathlib import Path

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

NEGATIVE = "Negative"
NEUTRAL = "Neutral"
POSITIVE = "Positive"

TOKEN_REGEX = re.compile(r"[a-zA-Z']+")
TOKEN_OR_PUNCT_REGEX = re.compile(r"[a-zA-Z']+|[.!?;,]")
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
}
CLAUSE_RESET_WORDS = {"but", "however", "though", "although", "yet"}


def ensure_vader_ready() -> None:
    nltk.download("vader_lexicon", quiet=True)


def vader_predict_labels(texts: list[str]) -> list[str]:
    ensure_vader_ready()
    analyzer = SentimentIntensityAnalyzer()

    labels: list[str] = []
    for text in texts:
        compound = analyzer.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            labels.append(POSITIVE)
        elif compound <= -0.05:
            labels.append(NEGATIVE)
        else:
            labels.append(NEUTRAL)
    return labels


def load_afinn(path: str | Path) -> dict[str, int]:
    lexicon: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            term, score = line.split("\t")
            lexicon[term.lower()] = int(score)
    return lexicon


def afinn_score_with_negation(text: str, lexicon: dict[str, int]) -> float:
    tokens = TOKEN_OR_PUNCT_REGEX.findall(str(text).lower())
    score = 0.0

    negation_active = False
    for token in tokens:
        if token in {".", "!", "?", ";", ","}:
            negation_active = False
            continue

        if token in CLAUSE_RESET_WORDS:
            negation_active = False
            continue

        if token in NEGATION_WORDS:
            negation_active = not negation_active
            continue

        if token in lexicon:
            token_score = lexicon[token]
            score += -token_score if negation_active else token_score

    return score


def afinn_predict_labels(texts: list[str], lexicon: dict[str, int]) -> list[str]:
    labels: list[str] = []
    for text in texts:
        score = afinn_score_with_negation(text, lexicon)
        if score > 0:
            labels.append(POSITIVE)
        elif score < 0:
            labels.append(NEGATIVE)
        else:
            labels.append(NEUTRAL)
    return labels
