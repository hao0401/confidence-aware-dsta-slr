import argparse
import shutil
from pathlib import Path

from experiment_specs import DATASET_SPECS, LITERATURE_BASELINES
from script_utils import (
    build_dataset_table_fieldnames,
    find_fusion_metrics,
    find_repo_root,
    PAPER_SINGLE_TOP1_FIELDS,
    write_csv,
    write_json,
)


ROOT = find_repo_root(__file__)
DEFAULT_OUT_DIR = ROOT / "paper_tables"


def load_metrics(dataset_name):
    return find_fusion_metrics(
        ROOT,
        f"conf_{dataset_name.lower().replace('-', '_')}",
        suffixes=("_fusion_results",),
    )


def format_percentage(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


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

    out_dir = Path(args.out_dir).resolve()
    write_csv(
        out_dir / "table_4_2_wlasl.csv",
        build_wlasl_rows(),
        build_dataset_table_fieldnames(
            ["WLASL100", "WLASL300", "WLASL1000", "WLASL2000"]
        ),
    )
    write_csv(
        out_dir / "table_4_2_msasl.csv",
        build_msasl_rows(),
        build_dataset_table_fieldnames(
            ["MSASL100", "MSASL200", "MSASL500", "MSASL1000"]
        ),
    )
    write_csv(
        out_dir / "table_4_2_slr500.csv",
        build_single_dataset_rows("SLR500"),
        PAPER_SINGLE_TOP1_FIELDS,
    )
    write_csv(
        out_dir / "table_4_2_nmfs_csl.csv",
        build_single_dataset_rows("NMFs-CSL"),
        PAPER_SINGLE_TOP1_FIELDS,
    )

    optional_files = {
        "table_4_3_ablation.csv": args.ablation_file,
        "table_4_3_hyperparams.csv": args.hyperparam_file,
        "table_4_4_robustness.csv": args.robustness_file,
    }
    for output_name, source in optional_files.items():
        if source and Path(source).exists():
            shutil.copy2(source, out_dir / output_name)

    write_json(out_dir / "datasets.json", DATASET_SPECS)


if __name__ == "__main__":
    main()
