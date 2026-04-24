from __future__ import annotations

import argparse
import json

from clean_social.labeling import sample_annotation_records, summarize_sampling
from clean_social.utils.io_utils import load_reviews_csv, save_dataframe
from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced 400-record annotation sheets.")
    parser.add_argument(
        "--input",
        default="data/raw/ali-express_reviews.csv",
        help="Input CSV path with original reviews.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/annotations",
        help="Directory where annotation files will be saved.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=400,
        help="Number of records to include in annotation set (required: 400).",
    )
    parser.add_argument(
        "--min-negation-ratio",
        type=float,
        default=0.10,
        help="Minimum negation ratio for Positive and Negative classes.",
    )
    parser.add_argument(
        "--sampling-label-source",
        choices=["vader", "rating"],
        default="vader",
        help="Label source used for balanced sampling.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    input_path = root / args.input
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews_csv(input_path, text_column="content")
    sampled = sample_annotation_records(
        df,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        text_column="content",
        score_column="score",
        min_negation_ratio=args.min_negation_ratio,
        sampling_label_source=args.sampling_label_source,
    )

    base_columns = [
        "record_id",
        "source_index",
        "userName",
        "score",
        "content",
        "at",
        "thumbsUpCount",
        "Sampling_Label",
        "Rating_Annotator",
        "has_negation",
    ]
    base_df = sampled[base_columns].copy()

    annotator2_df = base_df[["record_id", "content"]].copy()
    annotator2_df["Manual_Annotator2"] = ""

    annotator3_df = base_df[["record_id", "content"]].copy()
    annotator3_df["Manual_Annotator3"] = ""

    save_dataframe(base_df, output_dir / "review_base_400.csv")
    save_dataframe(annotator2_df, output_dir / "annotator2_sheet.csv")
    save_dataframe(annotator3_df, output_dir / "annotator3_sheet.csv")

    summary = summarize_sampling(base_df)
    print(json.dumps(summary, indent=2))

    print(f"Prepared {len(base_df)} records for annotation.")
    print(f"Files written to: {output_dir}")


if __name__ == "__main__":
    main()
