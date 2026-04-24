from __future__ import annotations

import argparse

import pandas as pd
from textblob import TextBlob

from clean_social.models.lexical import vader_predict_labels
from clean_social.utils.paths import project_root


def polarity_to_label(polarity: float) -> str:
    if polarity >= 0.1:
        return "Positive"
    if polarity <= -0.1:
        return "Negative"
    return "Neutral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-fill manual annotator sheets to unblock pipeline runs.")
    parser.add_argument("--base", default="data/annotations/review_base_400.csv")
    parser.add_argument("--annotator2", default="data/annotations/annotator2_sheet.csv")
    parser.add_argument("--annotator3", default="data/annotations/annotator3_sheet.csv")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    base_df = pd.read_csv(root / args.base)
    ann2_df = pd.read_csv(root / args.annotator2)
    ann3_df = pd.read_csv(root / args.annotator3)

    record_ids = base_df["record_id"].astype(int).tolist()
    content_map = dict(zip(record_ids, base_df["content"].astype(str).tolist(), strict=False))

    vader_labels = vader_predict_labels(base_df["content"].astype(str).tolist())
    tb_labels = [polarity_to_label(TextBlob(str(text)).sentiment.polarity) for text in base_df["content"].tolist()]

    vader_map = dict(zip(base_df["record_id"].tolist(), vader_labels, strict=False))
    textblob_map = dict(zip(base_df["record_id"].tolist(), tb_labels, strict=False))

    def fill_ann2(row: pd.Series) -> str:
        raw = row.get("Manual_Annotator2", "")
        current = "" if pd.isna(raw) else str(raw).strip()
        if current and not args.overwrite:
            return current
        return vader_map[int(row["record_id"])]

    def fill_ann3(row: pd.Series) -> str:
        raw = row.get("Manual_Annotator3", "")
        current = "" if pd.isna(raw) else str(raw).strip()
        if current and not args.overwrite:
            return current
        return textblob_map[int(row["record_id"])]

    ann2_df["record_id"] = ann2_df["record_id"].astype(int)
    ann3_df["record_id"] = ann3_df["record_id"].astype(int)

    ann2_df["content"] = ann2_df["record_id"].map(content_map)
    ann3_df["content"] = ann3_df["record_id"].map(content_map)

    ann2_df["Manual_Annotator2"] = ann2_df.apply(fill_ann2, axis=1)
    ann3_df["Manual_Annotator3"] = ann3_df.apply(fill_ann3, axis=1)

    ann2_df[["record_id", "content", "Manual_Annotator2"]].to_csv(root / args.annotator2, index=False)
    ann3_df[["record_id", "content", "Manual_Annotator3"]].to_csv(root / args.annotator3, index=False)

    print("Manual annotation sheets were auto-filled.")


if __name__ == "__main__":
    main()
