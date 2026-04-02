import argparse
import csv
import json
import shutil
from pathlib import Path

from experiment_specs import DATASET_SPECS, LITERATURE_BASELINES
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
DEFAULT_OUT_DIR = ROOT / "paper_tables"


def load_metrics(dataset_name):
    metrics_path = (
        ROOT
        / "work_dir"
        / f"conf_{dataset_name.lower().replace('-', '_')}_fusion_results"
        / "metrics.json"
    )
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def format_percentage(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_wlasl_rows():
    rows = list(LITERATURE_BASELINES["WLASL"])
    ours = {"method": "Ours"}
    for dataset_name in ["WLASL100", "WLASL300", "WLASL1000", "WLASL2000"]:
        metrics = load_metrics(dataset_name)
        ours[f"{dataset_name}_P-I"] = (
            None if metrics is None else format_percentage(metrics.get("top1"))
        )
        ours[f"{dataset_name}_P-C"] = (
            None
            if metrics is None
            else format_percentage(metrics.get("top1_per_class"))
        )
    rows.append(ours)
    return rows


def build_msasl_rows():
    rows = list(LITERATURE_BASELINES["MSASL"])
    ours = {"method": "Ours"}
    for dataset_name in ["MSASL100", "MSASL200", "MSASL500", "MSASL1000"]:
        metrics = load_metrics(dataset_name)
        ours[f"{dataset_name}_P-I"] = (
            None if metrics is None else format_percentage(metrics.get("top1"))
        )
        ours[f"{dataset_name}_P-C"] = (
            None
            if metrics is None
            else format_percentage(metrics.get("top1_per_class"))
        )
    rows.append(ours)
    return rows


def build_single_dataset_rows(dataset_key):
    rows = list(LITERATURE_BASELINES[dataset_key])
    dataset_name = "SLR500" if dataset_key == "SLR500" else "NMFs-CSL"
    metrics = load_metrics(dataset_name)
    rows.append(
        {
            "method": "Ours",
            "Top-1": None if metrics is None else format_percentage(metrics.get("top1")),
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export paper table CSV files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ablation-file")
    parser.add_argument("--hyperparam-file")
    parser.add_argument("--robustness-file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "table_4_2_wlasl.csv",
        [
            "method",
            "WLASL100_P-I",
            "WLASL100_P-C",
            "WLASL300_P-I",
            "WLASL300_P-C",
            "WLASL1000_P-I",
            "WLASL1000_P-C",
            "WLASL2000_P-I",
            "WLASL2000_P-C",
        ],
        build_wlasl_rows(),
    )
    write_csv(
        out_dir / "table_4_2_msasl.csv",
        [
            "method",
            "MSASL100_P-I",
            "MSASL100_P-C",
            "MSASL200_P-I",
            "MSASL200_P-C",
            "MSASL500_P-I",
            "MSASL500_P-C",
            "MSASL1000_P-I",
            "MSASL1000_P-C",
        ],
        build_msasl_rows(),
    )
    write_csv(
        out_dir / "table_4_2_slr500.csv",
        ["method", "Top-1"],
        build_single_dataset_rows("SLR500"),
    )
    write_csv(
        out_dir / "table_4_2_nmfs_csl.csv",
        ["method", "Top-1"],
        build_single_dataset_rows("NMFs-CSL"),
    )

    optional_files = {
        "table_4_3_ablation.csv": args.ablation_file,
        "table_4_3_hyperparams.csv": args.hyperparam_file,
        "table_4_4_robustness.csv": args.robustness_file,
    }
    for output_name, source in optional_files.items():
        if source and Path(source).exists():
            shutil.copy2(source, out_dir / output_name)

    with open(out_dir / "datasets.json", "w", encoding="utf-8") as handle:
        json.dump(DATASET_SPECS, handle, indent=2)


if __name__ == "__main__":
    main()
