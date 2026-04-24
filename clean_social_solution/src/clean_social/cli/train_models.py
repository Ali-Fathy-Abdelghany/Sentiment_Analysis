from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from clean_social.evaluation.metrics import compute_classification_metrics
from clean_social.models.lexical import afinn_predict_labels, load_afinn, vader_predict_labels
from clean_social.utils.paths import project_root

SCHEMES = ["s1", "s2", "s3"]
LABEL_TO_NUM = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model matrix and export evaluation metrics.")
    parser.add_argument("--features-dir", default="artifacts/features")
    parser.add_argument("--afinn", default="data/raw/AFINN-en-165.txt")
    parser.add_argument("--output", default="reports/model_metrics.csv")
    parser.add_argument("--details", default="reports/model_metrics_detailed.json")
    parser.add_argument("--models-dir", default="artifacts/models/benchmark")
    parser.add_argument("--plot-output", default="reports/plots/roc_auc_before_optimization.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def numeric_to_label(value: float) -> str:
    if value <= -0.33:
        return "Negative"
    if value >= 0.33:
        return "Positive"
    return "Neutral"


def load_feature_matrix(path: Path) -> tuple[list[int], np.ndarray]:
    df = pd.read_csv(path)
    record_ids = df["record_id"].astype(int).tolist()
    matrix = df.drop(columns=["record_id"]).to_numpy(dtype=float)
    return record_ids, matrix


def build_lr_score_matrix(pred_values: np.ndarray) -> np.ndarray:
    # Create class-wise ranking scores from the regression output for multiclass ROC-AUC.
    return np.column_stack((-pred_values, -np.abs(pred_values), pred_values))


def save_roc_auc_plot(report_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = report_df[
        report_df["representation"].isin(["tfidf", "glove"]) & report_df["roc_auc_ovr"].notna()
    ].copy()
    if plot_df.empty:
        return

    def _model_short(value: str) -> str:
        return "SVM" if value == "svm_linear" else "LR"

    plot_df["label"] = plot_df.apply(
        lambda row: f"{row['scheme']}-{row['representation']}-{_model_short(str(row['model']))}", axis=1
    )

    plt.figure(figsize=(14, 6))
    bars = plt.bar(plot_df["label"], plot_df["roc_auc_ovr"], color="#2f6f9f")
    plt.title("ROC-AUC (OVR) Before Optimization")
    plt.ylabel("ROC-AUC")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, plot_df["roc_auc_ovr"], strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.99, value + 0.01),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    root = project_root()

    features_dir = root / args.features_dir
    models_dir = root / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(features_dir / "labels_400.csv")

    y_map = dict(zip(labels_df["record_id"].astype(int), labels_df["ground_truth"], strict=False))
    record_ids = labels_df["record_id"].astype(int).tolist()
    y_all = [y_map[item] for item in record_ids]

    train_ids, test_ids = train_test_split(
        record_ids,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y_all,
    )

    afinn_lexicon = load_afinn(root / args.afinn)

    rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}
    model_manifest: dict[str, str] = {}

    for scheme in SCHEMES:
        texts_df = pd.read_csv(features_dir / f"texts_{scheme}.csv")
        text_map = dict(zip(texts_df["record_id"].astype(int), texts_df["content"].astype(str), strict=False))

        y_test = [y_map[item] for item in test_ids]
        x_test_texts = [text_map[item] for item in test_ids]

        vader_pred = vader_predict_labels(x_test_texts)
        vader_metrics = compute_classification_metrics(y_test, vader_pred)
        vader_key = f"{scheme}_vader"
        details[vader_key] = vader_metrics
        rows.append(
            {
                "scheme": scheme,
                "representation": "text",
                "model": "vader",
                **{k: v for k, v in vader_metrics.items() if k != "confusion_matrix"},
            }
        )

        afinn_pred = afinn_predict_labels(x_test_texts, afinn_lexicon)
        afinn_metrics = compute_classification_metrics(y_test, afinn_pred)
        afinn_key = f"{scheme}_afinn"
        details[afinn_key] = afinn_metrics
        rows.append(
            {
                "scheme": scheme,
                "representation": "text",
                "model": "afinn_negation",
                **{k: v for k, v in afinn_metrics.items() if k != "confusion_matrix"},
            }
        )

        for representation in ("tfidf", "glove"):
            feature_path = features_dir / f"{representation}_{scheme}.csv"
            ids, matrix = load_feature_matrix(feature_path)

            id_to_vec = {idx: vec for idx, vec in zip(ids, matrix, strict=False)}
            x_train = np.vstack([id_to_vec[item] for item in train_ids])
            x_test = np.vstack([id_to_vec[item] for item in test_ids])

            y_train = [y_map[item] for item in train_ids]

            svm_model = LinearSVC(random_state=args.random_seed)
            svm_model.fit(x_train, y_train)
            svm_pred = svm_model.predict(x_test).tolist()
            svm_scores = svm_model.decision_function(x_test)
            svm_metrics = compute_classification_metrics(y_test, svm_pred, score_matrix=np.asarray(svm_scores))
            svm_key = f"{scheme}_{representation}_svm"
            details[svm_key] = svm_metrics
            rows.append(
                {
                    "scheme": scheme,
                    "representation": representation,
                    "model": "svm_linear",
                    **{k: v for k, v in svm_metrics.items() if k != "confusion_matrix"},
                }
            )

            svm_path = models_dir / f"{scheme}_{representation}_svm_linear.joblib"
            joblib.dump(svm_model, svm_path)
            model_manifest[svm_key] = str(svm_path.relative_to(root)).replace("\\", "/")

            lr_model = LinearRegression()
            y_train_num = np.array([LABEL_TO_NUM[label] for label in y_train], dtype=float)
            lr_model.fit(x_train, y_train_num)

            lr_pred_num = lr_model.predict(x_test)
            lr_pred = [numeric_to_label(value) for value in lr_pred_num]
            lr_scores = build_lr_score_matrix(np.asarray(lr_pred_num))
            lr_metrics = compute_classification_metrics(y_test, lr_pred, score_matrix=lr_scores)
            lr_key = f"{scheme}_{representation}_linear_regression"
            details[lr_key] = lr_metrics
            rows.append(
                {
                    "scheme": scheme,
                    "representation": representation,
                    "model": "linear_regression",
                    **{k: v for k, v in lr_metrics.items() if k != "confusion_matrix"},
                }
            )

            lr_path = models_dir / f"{scheme}_{representation}_linear_regression.joblib"
            joblib.dump(lr_model, lr_path)
            model_manifest[lr_key] = str(lr_path.relative_to(root)).replace("\\", "/")

    details["saved_models"] = model_manifest

    report_df = pd.DataFrame(rows)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    details_path = root / args.details
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    save_roc_auc_plot(report_df, root / args.plot_output)

    print(f"Saved metrics table to {output_path}")
    print(f"Saved detailed metrics to {details_path}")
    print(f"Saved benchmark models to {models_dir}")
    print(f"Saved ROC-AUC plot to {root / args.plot_output}")


if __name__ == "__main__":
    main()
