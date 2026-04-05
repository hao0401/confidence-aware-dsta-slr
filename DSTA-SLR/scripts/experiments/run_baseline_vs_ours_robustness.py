import argparse
from pathlib import Path

from script_utils import (
    build_comparison_fieldnames,
    build_metric_comparison_rows,
    experiment_artifact_paths,
    find_fusion_metrics,
    find_repo_root,
    run_command,
    write_csv,
    write_json,
)
from python_locator import resolve_python


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)
STREAM = "joint"


def read_csv(path):
    import csv

    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_variant_robustness(label, prefix, device, num_worker, overwrite, out_dir):
    artifacts = experiment_artifact_paths(ROOT, f"{prefix}_{STREAM}")
    config_path = artifacts["config_path"]
    weights_path = artifacts["model_path"]
    robustness_path = out_dir / f"{label}_{STREAM}_robustness.csv"
    if overwrite or not robustness_path.exists():
        run_command(
            [
                PYTHON,
                "scripts/run_robustness_suite.py",
                "--config",
                str(config_path),
                "--weights",
                str(weights_path),
                "--device",
                str(device),
                "--num-worker",
                str(num_worker),
                "--out-file",
                str(robustness_path),
            ],
            cwd=ROOT,
        )
    return read_csv(robustness_path)


def safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def build_comparison_rows(baseline_rows, ours_rows):
    baseline_by_scenario = {row["scenario"]: row for row in baseline_rows}
    ours_by_scenario = {row["scenario"]: row for row in ours_rows}
    return build_metric_comparison_rows(
        baseline_by_scenario,
        ours_by_scenario,
        left_label="baseline",
        right_label="ours",
        value_transform=safe_float,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run joint-stream baseline vs ours robustness comparison."
    )
    parser.add_argument("--baseline-prefix", default="conf_wlasl100_43_baseline")
    parser.add_argument("--ours-prefix", default="conf_wlasl100")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "work_dir" / "baseline_vs_ours_robustness"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = run_variant_robustness(
        label="baseline",
        prefix=args.baseline_prefix,
        device=args.device,
        num_worker=args.num_worker,
        overwrite=args.overwrite_work_dir,
        out_dir=out_dir,
    )
    ours_rows = run_variant_robustness(
        label="ours",
        prefix=args.ours_prefix,
        device=args.device,
        num_worker=args.num_worker,
        overwrite=args.overwrite_work_dir,
        out_dir=out_dir,
    )
    comparison_rows = build_comparison_rows(baseline_rows, ours_rows)
    comparison_csv = out_dir / "comparison.csv"
    write_csv(
        comparison_csv,
        comparison_rows,
        build_comparison_fieldnames("baseline", "ours"),
    )

    payload = {
        "stream": STREAM,
        "baseline_prefix": args.baseline_prefix,
        "ours_prefix": args.ours_prefix,
        "baseline_clean_fusion": find_fusion_metrics(ROOT, args.baseline_prefix),
        "ours_clean_fusion": find_fusion_metrics(ROOT, args.ours_prefix),
        "comparison_rows": comparison_rows,
    }
    json_path = out_dir / "comparison.json"
    write_json(json_path, payload)

    print(comparison_csv)
    print(json_path)


if __name__ == "__main__":
    main()
