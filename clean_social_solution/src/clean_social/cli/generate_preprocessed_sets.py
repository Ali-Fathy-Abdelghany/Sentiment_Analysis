from __future__ import annotations

from pathlib import Path

from clean_social.preprocessing.pipeline import PreprocessingConfig, preprocess_dataframe
from clean_social.utils.io_utils import load_reviews_csv, save_dataframe
from clean_social.utils.paths import project_root

ROOT = project_root()
RAW_PATH = ROOT / "data" / "raw" / "ali-express_reviews.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def build_scheme_configs() -> dict[str, PreprocessingConfig]:
    """Define the 3 required preprocessing variants for downstream modeling."""
    return {
        "s1_minimal": PreprocessingConfig(
            lowercase=True,
            extract_tags=True,
        ),
        "s2_standard": PreprocessingConfig(
            lowercase=True,
            remove_urls=True,
            remove_emojis=True,
            lemmatize=True,
            extract_tags=True,
        ),
        "s3_extended": PreprocessingConfig(
            lowercase=True,
            remove_urls=True,
            remove_emojis=True,
            remove_punctuation=True,
            remove_stopwords=True,
            fix_spelling=True,
            lemmatize=True,
            extract_tags=True,
        ),
    }


def main() -> None:
    df = load_reviews_csv(RAW_PATH)

    if "source" not in df.columns:
        df["source"] = RAW_PATH.stem

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for scheme_name, config in build_scheme_configs().items():
        processed_df = preprocess_dataframe(df, config)
        output_path = PROCESSED_DIR / f"{scheme_name}.csv"
        save_dataframe(processed_df, output_path)
        print(f"{scheme_name}: {len(processed_df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
