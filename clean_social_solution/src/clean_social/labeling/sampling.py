from __future__ import annotations

import math
from typing import Any

import pandas as pd

from clean_social.labeling.rules import contains_negation, rating_to_sentiment
from clean_social.models.lexical import vader_predict_labels


def build_balanced_label_quotas(sample_size: int) -> dict[str, int]:
    base = sample_size // 3
    remainder = sample_size % 3
    quotas = {"Negative": base, "Positive": base, "Neutral": base}
    if remainder > 0:
        quotas["Neutral"] += 1
    if remainder > 1:
        quotas["Neutral"] += 1
    return quotas


def sample_annotation_records(
    df: pd.DataFrame,
    sample_size: int,
    random_seed: int,
    text_column: str = "content",
    score_column: str = "score",
    min_negation_ratio: float = 0.10,
    sampling_label_source: str = "vader",
) -> pd.DataFrame:
    if len(df) < sample_size:
        raise ValueError(f"Dataset has only {len(df)} rows; at least {sample_size} required.")

    working = df.copy()
    working["Rating_Annotator"] = working[score_column].apply(rating_to_sentiment)
    if sampling_label_source == "vader":
        working["Sampling_Label"] = vader_predict_labels(working[text_column].astype(str).tolist())
    elif sampling_label_source == "rating":
        working["Sampling_Label"] = working["Rating_Annotator"]
    else:
        raise ValueError("sampling_label_source must be 'vader' or 'rating'.")

    working["has_negation"] = working[text_column].astype(str).apply(contains_negation)

    quotas = build_balanced_label_quotas(sample_size)
    selected_frames: list[pd.DataFrame] = []

    for offset, label in enumerate(("Negative", "Neutral", "Positive")):
        label_pool = working[working["Sampling_Label"] == label]
        target_n = quotas[label]

        if len(label_pool) < target_n:
            raise ValueError(
                f"Not enough rows for {label}: required {target_n}, available {len(label_pool)}."
            )

        if label in {"Negative", "Positive"}:
            min_negation_n = int(math.ceil(target_n * min_negation_ratio))
            neg_pool = label_pool[label_pool["has_negation"]]
            if len(neg_pool) < min_negation_n:
                raise ValueError(
                    f"Negation quota unmet for {label}: required {min_negation_n}, available {len(neg_pool)}."
                )

            chosen_neg = neg_pool.sample(n=min_negation_n, random_state=random_seed + offset)
            remaining_pool = label_pool.drop(index=chosen_neg.index)
            remaining_n = target_n - min_negation_n
            chosen_rest = remaining_pool.sample(n=remaining_n, random_state=random_seed + 10 + offset)
            chosen = pd.concat([chosen_neg, chosen_rest], ignore_index=False)
        else:
            chosen = label_pool.sample(n=target_n, random_state=random_seed + offset)

        selected_frames.append(chosen)

    sampled = pd.concat(selected_frames, ignore_index=False)
    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(names="source_index")
    sampled["record_id"] = range(1, len(sampled) + 1)
    return sampled


def summarize_sampling(sampled_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": int(len(sampled_df)),
        "sampling_label_distribution": sampled_df["Sampling_Label"].value_counts().to_dict(),
        "rating_label_distribution": sampled_df["Rating_Annotator"].value_counts().to_dict(),
    }

    negation_stats = (
        sampled_df.groupby("Sampling_Label")["has_negation"].mean().fillna(0).mul(100).round(2).to_dict()
    )
    summary["negation_rate_percent"] = {k: float(v) for k, v in negation_stats.items()}
    return summary
