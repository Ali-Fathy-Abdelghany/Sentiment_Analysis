from __future__ import annotations

import argparse
from pathlib import Path

from clean_social.preprocessing.pipeline import PreprocessingConfig, preprocess_dataframe
from clean_social.utils.io_utils import load_reviews_csv, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configurable preprocessing pipeline for sentiment data.")

    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--text-column", default="content", help="Name of the text column.")
    parser.add_argument("--category-column", default="category", help="Name of generated category column.")

    # All flags default to False to satisfy task requirements.
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--remove-urls", action="store_true", default=False)
    parser.add_argument("--remove-emojis", action="store_true", default=False)
    parser.add_argument("--remove-punctuation", action="store_true", default=False)
    parser.add_argument("--remove-stopwords", action="store_true", default=False)
    parser.add_argument("--lemmatize", action="store_true", default=False)
    parser.add_argument("--fix-spelling", action="store_true", default=False)
    parser.add_argument("--extract-tags", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_reviews_csv(args.input, text_column=args.text_column)

    if "source" not in df.columns:
        df["source"] = Path(args.input).stem

    config = PreprocessingConfig(
        text_column=args.text_column,
        category_column=args.category_column,
        lowercase=args.lowercase,
        remove_urls=args.remove_urls,
        remove_emojis=args.remove_emojis,
        remove_punctuation=args.remove_punctuation,
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize,
        fix_spelling=args.fix_spelling,
        extract_tags=args.extract_tags,
    )

    output_df = preprocess_dataframe(df, config)
    save_dataframe(output_df, args.output)

    print(f"Saved {len(output_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
