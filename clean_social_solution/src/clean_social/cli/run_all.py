from __future__ import annotations

import argparse
import subprocess
import sys

PIPELINE_STEPS = [
    ("generate_preprocessed_sets", "clean_social.cli.generate_preprocessed_sets"),
    ("prepare_annotations", "clean_social.cli.prepare_annotations"),
    ("fill_annotations", "clean_social.cli.fill_annotations"),
    ("aggregate_annotations", "clean_social.cli.aggregate_annotations"),
    ("build_features", "clean_social.cli.build_features"),
    ("train_models", "clean_social.cli.train_models"),
    ("optimize_models", "clean_social.cli.optimize_models"),
    ("error_analysis", "clean_social.cli.error_analysis"),
    ("train_serving_model", "clean_social.cli.train_serving_model"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full CleanSocial pipeline in one command.")
    parser.add_argument(
        "--from-step",
        choices=[name for name, _ in PIPELINE_STEPS],
        default=PIPELINE_STEPS[0][0],
        help="Start execution from this pipeline step.",
    )
    parser.add_argument(
        "--to-step",
        choices=[name for name, _ in PIPELINE_STEPS],
        default=PIPELINE_STEPS[-1][0],
        help="Stop execution after this pipeline step.",
    )
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--min-negation-ratio", type=float, default=0.10)
    parser.add_argument("--sampling-label-source", choices=["vader", "rating"], default="vader")
    parser.add_argument(
        "--glove-path",
        default="auto",
        help="Path for local glove.6B.100d.txt passed to build_features.",
    )
    parser.add_argument(
        "--no-overwrite-annotations",
        action="store_true",
        default=False,
        help="Do not pass --overwrite to fill_annotations.",
    )
    return parser.parse_args()


def run_step(module_name: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, "-m", module_name]
    if extra_args:
        command.extend(extra_args)

    print(f"\n>>> Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    step_names = [name for name, _ in PIPELINE_STEPS]
    start_index = step_names.index(args.from_step)
    end_index = step_names.index(args.to_step)

    if start_index > end_index:
        raise ValueError("--from-step must come before --to-step in pipeline order.")

    for name, module_name in PIPELINE_STEPS[start_index : end_index + 1]:
        extra_args: list[str] = []
        if name == "prepare_annotations":
            extra_args = [
                "--sample-size",
                str(args.sample_size),
                "--min-negation-ratio",
                str(args.min_negation_ratio),
                "--sampling-label-source",
                args.sampling_label_source,
            ]
        elif name == "fill_annotations" and not args.no_overwrite_annotations:
            extra_args = ["--overwrite"]
        elif name == "build_features":
            extra_args = ["--glove-path", args.glove_path]

        run_step(module_name, extra_args)

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
