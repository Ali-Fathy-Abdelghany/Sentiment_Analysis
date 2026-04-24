from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from clean_social.apps.api import app
from clean_social.preprocessing.pipeline import PreprocessingConfig
from clean_social.utils.paths import project_root


def check_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def check_dir(path: Path) -> bool:
    return path.exists() and path.is_dir()


def bool_status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> None:
    root = project_root()
    reports_dir = root / "reports"

    raw_data = root / "data" / "raw" / "ali-express_reviews.csv"
    model_metrics = root / "reports" / "model_metrics.csv"
    optimization_metrics = root / "reports" / "optimization_results.csv"
    deployment_model = root / "artifacts" / "models" / "deployment_model.joblib"
    agreement_report = root / "reports" / "annotation_agreement.json"

    checks: list[tuple[str, bool, str]] = []

    # Task 1
    t1_ok = check_file(raw_data)
    records_ok = False
    if t1_ok:
        df_raw = pd.read_csv(raw_data)
        records_ok = len(df_raw) >= 100
    checks.append(("Task 1 raw dataset exists", t1_ok, str(raw_data)))
    checks.append(("Task 1 minimum records >= 100", records_ok, "data/raw/ali-express_reviews.csv"))

    # Task 2
    cfg = PreprocessingConfig()
    defaults_false = not any(
        [
            cfg.lowercase,
            cfg.remove_urls,
            cfg.remove_emojis,
            cfg.remove_punctuation,
            cfg.remove_stopwords,
            cfg.lemmatize,
            cfg.fix_spelling,
            cfg.extract_tags,
        ]
    )
    checks.append(
        (
            "Task 2 configurable preprocessing CLI module exists",
            check_file(root / "src" / "clean_social" / "cli" / "preprocess.py"),
            "src/clean_social/cli/preprocess.py",
        )
    )
    checks.append(
        (
            "Task 2 main.py entrypoint module exists",
            check_file(root / "src" / "clean_social" / "cli" / "main.py"),
            "src/clean_social/cli/main.py",
        )
    )
    checks.append(("Task 2 default cleaning flags are False", defaults_false, "PreprocessingConfig"))

    # Task 3
    task3_files_ok = all(
        [
            check_file(root / "data" / "annotations" / "labels_400.csv"),
            check_file(agreement_report),
            check_file(root / "artifacts" / "features" / "tfidf_s1.csv"),
            check_file(root / "artifacts" / "features" / "glove_s1.csv"),
            check_file(model_metrics),
        ]
    )

    metrics_rows_ok = False
    if check_file(model_metrics):
        df_metrics = pd.read_csv(model_metrics)
        metrics_rows_ok = len(df_metrics) == 18

    checks.append(("Task 3 labeling/features/metrics artifacts exist", task3_files_ok, "data/, artifacts/, reports/"))
    checks.append(("Task 3 benchmark includes 18 evaluated model variants", metrics_rows_ok, "reports/model_metrics.csv"))

    # Task 4
    t4_files_ok = all(
        [
            check_file(optimization_metrics),
            check_file(root / "reports" / "error_analysis.md"),
            check_file(deployment_model),
            check_file(root / "src" / "clean_social" / "apps" / "api.py"),
            check_file(root / "src" / "clean_social" / "apps" / "streamlit_app.py"),
        ]
    )
    checks.append(("Task 4 optimization/error/deployment files exist", t4_files_ok, "reports/, artifacts/, src/clean_social/apps/"))

    api_predict_ok = False
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "I love this product"})
        api_predict_ok = response.status_code == 200 and "sentiment" in response.json()
    except Exception:
        api_predict_ok = False

    checks.append(("Task 4 API predict endpoint works", api_predict_ok, "POST /predict"))

    lines = ["# Task Coverage Check", ""]
    pass_count = 0
    for title, passed, detail in checks:
        status = bool_status(passed)
        if passed:
            pass_count += 1
        lines.append(f"- [{status}] {title} ({detail})")

    lines.append("")
    lines.append(f"Summary: {pass_count}/{len(checks)} checks passed.")

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_md = reports_dir / "task_coverage_check.md"
    report_json = reports_dir / "task_coverage_check.json"

    report_md.write_text("\n".join(lines), encoding="utf-8")
    report_json.write_text(
        json.dumps(
            {
                "passed": pass_count,
                "total": len(checks),
                "checks": [
                    {"title": title, "passed": passed, "detail": detail}
                    for title, passed, detail in checks
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nSaved: {report_md}")
    print(f"Saved: {report_json}")


if __name__ == "__main__":
    main()
