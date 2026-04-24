from __future__ import annotations

import argparse
import json

import pandas as pd

from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data collection audit summary.")
    parser.add_argument("--input", default="data/raw/ali-express_reviews.csv")
    parser.add_argument("--output-json", default="reports/data_audit.json")
    parser.add_argument("--output-md", default="reports/data_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    df = pd.read_csv(root / args.input)

    summary = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "null_counts": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "score_distribution": {str(k): int(v) for k, v in df["score"].value_counts().sort_index().to_dict().items()},
        "min_required_records": 100,
        "passes_min_requirement": bool(len(df) >= 100),
    }

    out_json = root / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Data Audit",
        "",
        f"- Total records: {summary['rows']}",
        f"- Minimum required records: {summary['min_required_records']}",
        f"- Passes requirement: {summary['passes_min_requirement']}",
        f"- Duplicate rows: {summary['duplicate_rows']}",
        "",
        "## Score Distribution",
    ]

    for score, count in summary["score_distribution"].items():
        lines.append(f"- Score {score}: {count}")

    (root / args.output_md).write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved audit JSON to {out_json}")
    print(f"Saved audit markdown to {root / args.output_md}")


if __name__ == "__main__":
    main()
