from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CategoryPattern:
    label: str
    pattern: str


DEFAULT_CATEGORY_PATTERNS: tuple[CategoryPattern, ...] = (
    CategoryPattern("delivery_shipping", r"\b(ship|shipping|deliver|delivery|courier|tracking|arriv|package)\b"),
    CategoryPattern("refund_return", r"\b(refund|return|chargeback|cancel|compensation|reimburse)\b"),
    CategoryPattern("product_quality", r"\b(fake|defect|broken|damaged|quality|counterfeit|faulty)\b"),
    CategoryPattern("pricing_discount", r"\b(price|cost|expensive|cheap|discount|deal|tax|fee|charge)\b"),
    CategoryPattern("app_usability", r"\b(app|crash|bug|error|login|sign\s?in|network|update|slow|lag)\b"),
    CategoryPattern("customer_service", r"\b(support|service|help\s?center|agent|response|reply)\b"),
    CategoryPattern("ads_promotions", r"\b(ad|spam|notification|promo|promotion|email)\b"),
    CategoryPattern("scam_trust", r"\b(scam|fraud|dishonest|thief|stole|trust)\b"),
)


class CategoryTagger:
    """Regex-based single-label categorization for review metadata tagging."""

    def __init__(self, patterns: Iterable[CategoryPattern] = DEFAULT_CATEGORY_PATTERNS) -> None:
        self._compiled = [(item.label, re.compile(item.pattern, flags=re.IGNORECASE)) for item in patterns]

    def predict(self, text: str) -> str:
        value = str(text)
        for label, pattern in self._compiled:
            if pattern.search(value):
                return label
        return "other"
