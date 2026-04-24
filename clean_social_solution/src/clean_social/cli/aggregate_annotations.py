from __future__ import annotations

import argparse
import json

import pandas as pd

from clean_social.labeling import aggregate_annotations, compute_kappa_report
from clean_social.utils.io_utils import ensure_required_columns, load_reviews_csv, save_dataframe
from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 3 annotators using majority voting and Cohen's Kappa.")
    parser.add_argument("--base", default="data/annotations/review_base_400.csv")
    parser.add_argument("--annotator2", default="data/annotations/annotator2_sheet.csv")
    parser.add_argument("--annotator3", default="data/annotations/annotator3_sheet.csv")
    parser.add_argument("--output", default="data/annotations/labels_400.csv")
    parser.add_argument("--report", default="reports/annotation_agreement.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    base_df = load_reviews_csv(root / args.base, text_column="content")
    annotator2_df = pd.read_csv(root / args.annotator2)
    annotator3_df = pd.read_csv(root / args.annotator3)

    ensure_required_columns(base_df, ["record_id", "Rating_Annotator"])
    ensure_required_columns(annotator2_df, ["record_id", "Manual_Annotator2"])
    ensure_required_columns(annotator3_df, ["record_id", "Manual_Annotator3"])

    merged = aggregate_annotations(base_df, annotator2_df, annotator3_df)
    kappa_report = compute_kappa_report(merged)

    save_dataframe(merged, root / args.output)
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(kappa_report, indent=2), encoding="utf-8")

    print("Saved final labeled dataset and kappa report.")
    print(json.dumps(kappa_report, indent=2))


if __name__ == "__main__":
    main()
