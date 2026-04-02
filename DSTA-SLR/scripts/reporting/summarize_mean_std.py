import argparse
import csv
import json
import statistics
from pathlib import Path
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)


def load_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_mean_std(values):
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0] * 100:.2f}"
    mean = statistics.mean(values) * 100
    std = statistics.stdev(values) * 100
    return f"{mean:.2f} ± {std:.2f}"


def collect_seed_metrics(prefix, metric_path):
    values = []
    for seed in range(1, 6):
        path = ROOT / "work_dir" / f"{prefix}_seed{seed}" / metric_path
        if path.exists():
            values.append(load_metrics(path))
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
        row = {
            "variant": variant,
            "joint_top1_mean_std": format_mean_std(
                [entry["top1"] for entry in joint_metrics]
            ),
            "fusion_top1_mean_std": format_mean_std(
                [entry["top1"] for entry in fusion_metrics]
            ),
            "fusion_pc_mean_std": format_mean_std(
                [entry["top1_per_class"] for entry in fusion_metrics]
            ),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize mean ± std results for WLASL100 repeated runs."
    )
    parser.add_argument(
        "--out-file",
        default=str(ROOT / "work_dir" / "paper_tables_wlasl100" / "table_4_3_mean_std.csv"),
    )
    args = parser.parse_args()

    rows = summarize_wlasl100()
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "joint_top1_mean_std",
                "fusion_top1_mean_std",
                "fusion_pc_mean_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
