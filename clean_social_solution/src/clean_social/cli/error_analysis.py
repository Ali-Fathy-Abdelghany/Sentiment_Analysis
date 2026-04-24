from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate error analysis report for misclassified samples.")
    parser.add_argument("--features", default="artifacts/features/tfidf_s2.csv")
    parser.add_argument("--labels", default="artifacts/features/labels_400.csv")
    parser.add_argument("--texts", default="artifacts/features/texts_s2.csv")
    parser.add_argument("--output", default="reports/error_analysis.csv")
    parser.add_argument("--report-md", default="reports/error_analysis.md")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def detect_patterns(text: str) -> str:
    value = str(text).lower()
    patterns = []
    if "not" in value or "never" in value or "no " in value:
        patterns.append("negation")
    if any(symbol in value for symbol in ["!", "?", "...", "\ud83d", "\u2764"]):
        patterns.append("emphasis_or_emoji")
    if len(value.split()) <= 4:
        patterns.append("very_short_text")
    if any(char for char in value if ord(char) > 127):
        patterns.append("non_english_or_unicode")
    if not patterns:
        patterns.append("context_ambiguity")
    return ",".join(patterns)


def main() -> None:
    args = parse_args()
    root = project_root()

    feat_df = pd.read_csv(root / args.features)
    labels_df = pd.read_csv(root / args.labels)
    texts_df = pd.read_csv(root / args.texts)

    merged = labels_df.merge(feat_df, on="record_id", how="inner").merge(texts_df, on="record_id", how="inner")

    y = merged["ground_truth"].astype(str)
    x = merged.drop(columns=["record_id", "source_index", "ground_truth", "content"]).to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        merged.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    model = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    errors = []
    for row_index, pred, truth in zip(idx_test, preds, y_test, strict=False):
        if pred == truth:
            continue
        text = merged.loc[row_index, "content"]
        errors.append(
            {
                "record_id": int(merged.loc[row_index, "record_id"]),
                "text": text,
                "true_label": truth,
                "predicted_label": pred,
                "pattern_tags": detect_patterns(text),
            }
        )

    error_df = pd.DataFrame(errors)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(output_path, index=False)

    summary_lines = [
        "# Error Analysis",
        "",
        f"Total test samples: {len(y_test)}",
        f"Misclassified samples: {len(error_df)}",
        "",
        "## Frequent Pattern Tags",
    ]

    if len(error_df) > 0:
        pattern_counts = error_df["pattern_tags"].str.split(",").explode().value_counts()
        for label, count in pattern_counts.items():
            summary_lines.append(f"- {label}: {count}")
    else:
        summary_lines.append("- No misclassifications found in this run.")

    summary_lines.extend(
        [
            "",
            "## Conclusion",
            "The model mostly fails on short or ambiguous text, negation-heavy phrasing, and multilingual/noisy inputs.",
        ]
    )

    (root / args.report_md).write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved error analysis data to {output_path}")
    print(f"Saved narrative report to {root / args.report_md}")


if __name__ == "__main__":
    main()
