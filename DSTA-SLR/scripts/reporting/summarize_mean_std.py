import argparse
from pathlib import Path

from script_utils import (
    find_repo_root,
    read_json,
    summarize_metric_series,
    WLASL100_MEAN_STD_FIELDS,
    write_csv,
)


ROOT = find_repo_root(__file__)


def collect_seed_metrics(prefix, metric_path):
    values = []
    for seed in range(1, 6):
        path = ROOT / "work_dir" / f"{prefix}_seed{seed}" / metric_path
        if path.exists():
            values.append(read_json(path))
    return values


def summarize_wlasl100():
    variants = {
        "baseline": "conf_wlasl100_43_baseline",
        "node_encoding_only": "conf_wlasl100_43_node_encoding_only",
        "graph_only": "conf_wlasl100_43_graph_only",
        "temporal_only": "conf_wlasl100_43_temporal_only",
        "all_modules": "conf_wlasl100_43_all_modules",
    }
    rows = []
    for variant, prefix in variants.items():
        joint_metrics = collect_seed_metrics(
            f"{prefix}_joint", Path("eval_results") / "best_metrics.json"
        )
        fusion_metrics = collect_seed_metrics(
            f"{prefix}_fusion", Path("metrics.json")
        )
        row = {"variant": variant}
        row.update(
            summarize_metric_series(
                {
                    "joint_top1_mean_std": [entry["top1"] for entry in joint_metrics],
                    "fusion_top1_mean_std": [entry["top1"] for entry in fusion_metrics],
                    "fusion_pc_mean_std": [
                        entry["top1_per_class"] for entry in fusion_metrics
                    ],
                },
                separator=" +/- ",
            )
        )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize mean +/- std results for WLASL100 repeated runs."
    )
    parser.add_argument(
        "--out-file",
        default=str(ROOT / "work_dir" / "paper_tables_wlasl100" / "table_4_3_mean_std.csv"),
    )
    args = parser.parse_args()

    rows = summarize_wlasl100()
    write_csv(
        Path(args.out_file).resolve(),
        rows,
        WLASL100_MEAN_STD_FIELDS,
    )


if __name__ == "__main__":
    main()
