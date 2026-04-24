from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from clean_social.labeling.rules import normalize_label


def majority_vote(rating_label: str, annotator2: str, annotator3: str) -> str:
    labels = [rating_label, annotator2, annotator3]
    counts = Counter(labels)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]

    # A 3-way tie occurs when all labels differ; choose rating-based label deterministically.
    if len(winners) > 1:
        return rating_label
    return winners[0]


def aggregate_annotations(base_df: pd.DataFrame, annotator2_df: pd.DataFrame, annotator3_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(annotator2_df[["record_id", "Manual_Annotator2"]], on="record_id", how="inner")
    merged = merged.merge(annotator3_df[["record_id", "Manual_Annotator3"]], on="record_id", how="inner")

    merged["Rating_Annotator"] = merged["Rating_Annotator"].apply(normalize_label)
    merged["Manual_Annotator2"] = merged["Manual_Annotator2"].apply(normalize_label)
    merged["Manual_Annotator3"] = merged["Manual_Annotator3"].apply(normalize_label)

    merged["ground_truth"] = merged.apply(
        lambda row: majority_vote(
            row["Rating_Annotator"], row["Manual_Annotator2"], row["Manual_Annotator3"]
        ),
        axis=1,
    )
    return merged


def compute_kappa_report(merged_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "records": int(len(merged_df)),
        "kappa_rating_vs_annotator2": float(
            cohen_kappa_score(merged_df["Rating_Annotator"], merged_df["Manual_Annotator2"])
        ),
        "kappa_rating_vs_annotator3": float(
            cohen_kappa_score(merged_df["Rating_Annotator"], merged_df["Manual_Annotator3"])
        ),
        "kappa_annotator2_vs_annotator3": float(
            cohen_kappa_score(merged_df["Manual_Annotator2"], merged_df["Manual_Annotator3"])
        ),
    }
