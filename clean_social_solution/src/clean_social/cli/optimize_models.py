from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from clean_social.evaluation.metrics import compute_classification_metrics
from clean_social.utils.paths import project_root

SCHEMES = ["s1", "s2", "s3"]
REPRS = ["tfidf", "glove"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model optimization and compare before/after metrics.")
    parser.add_argument("--features-dir", default="artifacts/features")
    parser.add_argument("--output-csv", default="reports/optimization_results.csv")
    parser.add_argument("--output-json", default="reports/optimization_results_detailed.json")
    parser.add_argument("--models-dir", default="artifacts/models/optimized")
    parser.add_argument("--plot-output", default="reports/plots/roc_auc_after_optimization.png")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def load_features(path: Path) -> tuple[list[int], np.ndarray]:
    df = pd.read_csv(path)
    return df["record_id"].astype(int).tolist(), df.drop(columns=["record_id"]).to_numpy(dtype=float)


def save_comparison_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty:
        return

    labels = results_df.apply(lambda row: f"{row['scheme']}-{row['representation']}", axis=1).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].bar(x - width / 2, results_df["baseline_roc_auc"], width, label="Baseline", color="#8fa4b7")
    axes[0].bar(x + width / 2, results_df["tuned_roc_auc"], width, label="Tuned", color="#2f6f9f")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("ROC-AUC (OVR)")
    axes[0].set_title("ROC-AUC Before vs After Optimization")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, results_df["baseline_f1"], width, label="Baseline", color="#c8b792")
    axes[1].bar(x + width / 2, results_df["tuned_f1"], width, label="Tuned", color="#8f6a2f")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("F1 Macro")
    axes[1].set_title("F1 Macro Before vs After Optimization")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    features_dir = root / args.features_dir
    models_dir = root / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(features_dir / "labels_400.csv")
    labels_map = dict(zip(labels_df["record_id"].astype(int), labels_df["ground_truth"], strict=False))

    optimization_rows: list[dict[str, object]] = []
    optimization_details: dict[str, dict[str, object]] = {}

    random_chance_f1 = 1.0 / 3.0
    required_f1 = random_chance_f1 * 1.2

    for scheme in SCHEMES:
        for representation in REPRS:
            ids, matrix = load_features(features_dir / f"{representation}_{scheme}.csv")
            y = np.array([labels_map[item] for item in ids])

            x_train, x_test, y_train, y_test = train_test_split(
                matrix,
                y,
                test_size=args.test_size,
                random_state=args.random_seed,
                stratify=y,
            )

            baseline_model = SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=args.random_seed,
            )
            baseline_model.fit(x_train, y_train)
            baseline_pred = baseline_model.predict(x_test)
            baseline_scores = baseline_model.predict_proba(x_test)
            baseline_metrics = compute_classification_metrics(
                y_test.tolist(), baseline_pred.tolist(), score_matrix=baseline_scores
            )

            grid = GridSearchCV(
                estimator=SVC(probability=True, class_weight="balanced", random_state=args.random_seed),
                param_grid={
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1.0, 3.0, 10.0],
                    "gamma": ["scale", "auto"],
                },
                scoring="f1_macro",
                cv=5,
                n_jobs=-1,
            )
            grid.fit(x_train, y_train)

            tuned_model = grid.best_estimator_
            tuned_pred = tuned_model.predict(x_test)
            tuned_scores = tuned_model.predict_proba(x_test)
            tuned_metrics = compute_classification_metrics(
                y_test.tolist(), tuned_pred.tolist(), score_matrix=tuned_scores
            )

            model_key = f"{scheme}_{representation}_svc"
            model_path = models_dir / f"{model_key}.joblib"
            joblib.dump(tuned_model, model_path)

            optimization_details[model_key] = {
                "best_params": grid.best_params_,
                "model_path": str(model_path.relative_to(root)).replace("\\", "/"),
                "baseline": baseline_metrics,
                "tuned": tuned_metrics,
                "required_f1_threshold": required_f1,
            }

            optimization_rows.append(
                {
                    "scheme": scheme,
                    "representation": representation,
                    "baseline_f1": baseline_metrics["f1_macro"],
                    "tuned_f1": tuned_metrics["f1_macro"],
                    "f1_delta": tuned_metrics["f1_macro"] - baseline_metrics["f1_macro"],
                    "baseline_roc_auc": baseline_metrics["roc_auc_ovr"],
                    "tuned_roc_auc": tuned_metrics["roc_auc_ovr"],
                    "roc_auc_delta": (
                        tuned_metrics["roc_auc_ovr"] - baseline_metrics["roc_auc_ovr"]
                        if tuned_metrics["roc_auc_ovr"] is not None and baseline_metrics["roc_auc_ovr"] is not None
                        else None
                    ),
                    "required_f1_threshold": required_f1,
                    "passes_requirement": tuned_metrics["f1_macro"] >= required_f1,
                    "best_params": json.dumps(grid.best_params_),
                }
            )

    output_df = pd.DataFrame(optimization_rows)
    out_csv = root / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_csv, index=False)

    out_json = root / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(optimization_details, indent=2), encoding="utf-8")

    save_comparison_plot(output_df, root / args.plot_output)

    print(f"Saved optimization summary to {out_csv}")
    print(f"Saved optimization details to {out_json}")
    print(f"Saved tuned models to {models_dir}")
    print(f"Saved optimization plot to {root / args.plot_output}")


if __name__ == "__main__":
    main()
