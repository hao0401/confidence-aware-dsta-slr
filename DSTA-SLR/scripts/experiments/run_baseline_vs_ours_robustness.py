import argparse
import json
import sys
from pathlib import Path


from script_utils import find_repo_root, read_json, run_command, write_csv


ROOT = find_repo_root(__file__)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from python_locator import resolve_python

PYTHON = resolve_python(ROOT)
STREAM = "joint"
FUSION_DIR_SUFFIXES = ("_fusion_results", "_fusion")


def read_csv(path):
    import csv

    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_fusion_metrics(prefix):
    for suffix in FUSION_DIR_SUFFIXES:
        metrics_path = ROOT / "work_dir" / f"{prefix}{suffix}" / "metrics.json"
        if metrics_path.exists():
            return read_json(metrics_path)
    return None


def run_variant_robustness(label, prefix, device, num_worker, overwrite, out_dir):
    experiment_dir = ROOT / "work_dir" / f"{prefix}_{STREAM}"
    config_path = experiment_dir / "config.yaml"
    weights_path = experiment_dir / "save_models" / "best_model.pt"
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
    scenarios = list(dict.fromkeys([*baseline_by_scenario.keys(), *ours_by_scenario.keys()]))

    clean_baseline_top1 = safe_float(baseline_by_scenario.get("clean", {}).get("top1"))
    clean_ours_top1 = safe_float(ours_by_scenario.get("clean", {}).get("top1"))
    rows = []
    for scenario in scenarios:
        baseline_row = baseline_by_scenario.get(scenario, {})
        ours_row = ours_by_scenario.get(scenario, {})
        baseline_top1 = safe_float(baseline_row.get("top1"))
        ours_top1 = safe_float(ours_row.get("top1"))
        baseline_top5 = safe_float(baseline_row.get("top5"))
        ours_top5 = safe_float(ours_row.get("top5"))
        rows.append(
            {
                "scenario": scenario,
                "baseline_top1": baseline_top1,
                "ours_top1": ours_top1,
                "top1_gain": None
                if baseline_top1 is None or ours_top1 is None
                else ours_top1 - baseline_top1,
                "baseline_top1_retention": None
                if baseline_top1 is None or clean_baseline_top1 in (None, 0.0)
                else baseline_top1 / clean_baseline_top1,
                "ours_top1_retention": None
                if ours_top1 is None or clean_ours_top1 in (None, 0.0)
                else ours_top1 / clean_ours_top1,
                "baseline_top5": baseline_top5,
                "ours_top5": ours_top5,
                "top5_gain": None
                if baseline_top5 is None or ours_top5 is None
                else ours_top5 - baseline_top5,
            }
        )
    return rows


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

    out_dir = Path(args.out_dir)
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
        [
            "scenario",
            "baseline_top1",
            "ours_top1",
            "top1_gain",
            "baseline_top1_retention",
            "ours_top1_retention",
            "baseline_top5",
            "ours_top5",
            "top5_gain",
        ],
    )

    payload = {
        "stream": STREAM,
        "baseline_prefix": args.baseline_prefix,
        "ours_prefix": args.ours_prefix,
        "baseline_clean_fusion": find_fusion_metrics(args.baseline_prefix),
        "ours_clean_fusion": find_fusion_metrics(args.ours_prefix),
        "comparison_rows": comparison_rows,
    }
    json_path = out_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(comparison_csv)
    print(json_path)


if __name__ == "__main__":
    main()
