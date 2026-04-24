from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clean_social.features.representations import build_glove_matrix, build_tfidf_matrix
from clean_social.utils.io_utils import save_dataframe
from clean_social.utils.paths import project_root

SCHEME_FILES = {
    "s1": "data/processed/s1_minimal.csv",
    "s2": "data/processed/s2_standard.csv",
    "s3": "data/processed/s3_extended.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TF-IDF and GloVe features for labeled records.")
    parser.add_argument("--labels", default="data/annotations/labels_400.csv")
    parser.add_argument("--output-dir", default="artifacts/features")
    parser.add_argument("--tfidf-max-features", type=int, default=3000)
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument(
        "--glove-path",
        default="auto",
        help="Path to local GloVe txt file (use 'auto' to discover common locations).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    glove_path: Path | None = None
    if str(args.glove_path).lower() != "auto":
        candidate = Path(args.glove_path)
        glove_path = candidate if candidate.is_absolute() else (root / candidate)
    else:
        auto_candidates = [
            root / "data" / "raw" / "glove.6B.100d.txt",
            root.parent / "glove.6B.100d.txt",
        ]
        glove_path = next((path for path in auto_candidates if path.exists()), None)

    labels_df = pd.read_csv(root / args.labels)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_export = labels_df[["record_id", "source_index", "ground_truth"]].copy()
    save_dataframe(labels_export, output_dir / "labels_400.csv")

    metadata: dict[str, dict[str, str | int]] = {}

    for scheme_name, relative_path in SCHEME_FILES.items():
        scheme_df = pd.read_csv(root / relative_path).reset_index(names="source_index")
        subset = labels_df[["record_id", "source_index", "ground_truth"]].merge(
            scheme_df[["source_index", "content"]], on="source_index", how="inner"
        )

        texts = subset["content"].astype(str).tolist()

        tfidf_bundle = build_tfidf_matrix(texts, max_features=args.tfidf_max_features)
        tfidf_df = pd.DataFrame(tfidf_bundle.matrix, columns=tfidf_bundle.feature_names)
        tfidf_df.insert(0, "record_id", subset["record_id"].values)
        save_dataframe(tfidf_df, output_dir / f"tfidf_{scheme_name}.csv")

        glove_bundle = build_glove_matrix(texts, embedding_dim=args.glove_dim, local_glove_path=glove_path)
        glove_columns = [f"g{i}" for i in range(glove_bundle.embedding_dim)]
        glove_df = pd.DataFrame(glove_bundle.matrix, columns=glove_columns)
        glove_df.insert(0, "record_id", subset["record_id"].values)
        save_dataframe(glove_df, output_dir / f"glove_{scheme_name}.csv")

        save_dataframe(subset[["record_id", "content"]], output_dir / f"texts_{scheme_name}.csv")

        metadata[scheme_name] = {
            "rows": int(len(subset)),
            "tfidf_features": int(len(tfidf_bundle.feature_names)),
            "glove_dim": int(glove_bundle.embedding_dim),
            "glove_source": glove_bundle.source,
        }

    (output_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
