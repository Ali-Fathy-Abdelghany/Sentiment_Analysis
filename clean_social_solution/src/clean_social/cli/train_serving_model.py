from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save deployable sentiment model artifact.")
    parser.add_argument("--labels", default="artifacts/features/labels_400.csv")
    parser.add_argument("--texts", default="artifacts/features/texts_s2.csv")
    parser.add_argument("--output-model", default="artifacts/models/deployment_model.joblib")
    parser.add_argument("--output-metrics", default="reports/serving_model_metrics.json")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    labels_df = pd.read_csv(root / args.labels)
    texts_df = pd.read_csv(root / args.texts)

    merged = labels_df.merge(texts_df, on="record_id", how="inner")
    merged["content"] = merged["content"].fillna("")
    x = merged["content"].astype(str).tolist()
    y = merged["ground_truth"].astype(str).tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            (
                "svm",
                SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "labels": sorted(set(y)),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }

    model_path = root / args.output_model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = root / args.output_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved deployment model to {model_path}")
    print(f"Saved deployment metrics to {metrics_path}")


if __name__ == "__main__":
    main()
