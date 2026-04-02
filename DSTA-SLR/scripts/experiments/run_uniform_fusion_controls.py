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

VARIANT_STREAM_DIRS = {
    "baseline": {
        "joint": "conf_wlasl100_43_baseline_joint",
        "bone": "conf_wlasl100_43_baseline_bone",
        "joint_motion": "conf_wlasl100_43_baseline_joint_motion",
        "bone_motion": "conf_wlasl100_43_baseline_bone_motion",
    },
    "node_encoding_only": {
        "joint": "conf_wlasl100_43_node_encoding_only_joint",
        "bone": "conf_wlasl100_43_node_encoding_only_bone",
        "joint_motion": "conf_wlasl100_43_node_encoding_only_joint_motion",
        "bone_motion": "conf_wlasl100_43_node_encoding_only_bone_motion",
    },
    "graph_only": {
        "joint": "conf_wlasl100_43_graph_only_joint",
        "bone": "conf_wlasl100_43_graph_only_bone",
        "joint_motion": "conf_wlasl100_43_graph_only_joint_motion",
        "bone_motion": "conf_wlasl100_43_graph_only_bone_motion",
    },
    "temporal_only": {
        "joint": "conf_wlasl100_43_temporal_only_joint",
        "bone": "conf_wlasl100_43_temporal_only_bone",
        "joint_motion": "conf_wlasl100_43_temporal_only_joint_motion",
        "bone_motion": "conf_wlasl100_43_temporal_only_bone_motion",
    },
    "full_model": {
        "joint": "conf_wlasl100_joint",
        "bone": "conf_wlasl100_bone",
        "joint_motion": "conf_wlasl100_joint_motion",
        "bone_motion": "conf_wlasl100_bone_motion",
    },
    "consistency": {
        "joint": "conf_wlasl100_consistency_joint",
        "bone": "conf_wlasl100_consistency_bone",
        "joint_motion": "conf_wlasl100_consistency_joint_motion",
        "bone_motion": "conf_wlasl100_consistency_bone_motion",
    },
}
def main():
    parser = argparse.ArgumentParser(
        description="Run uniform-fusion controls for WLASL100 ablation variants."
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=tuple(VARIANT_STREAM_DIRS.keys()),
        default=["baseline", "node_encoding_only", "graph_only", "temporal_only", "full_model"],
    )
    parser.add_argument(
        "--label-path",
        default=str(ROOT / "data" / "WLASL100" / "val_label.pkl"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "work_dir" / "uniform_fusion_controls"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in args.variants:
        stream_dirs = VARIANT_STREAM_DIRS[variant]
        variant_out = out_dir / variant
        run_command(
            [
                PYTHON,
                "ensemble/fuse_streams_uniform.py",
                "--label-path",
                args.label_path,
                "--joint",
                str(ROOT / "work_dir" / stream_dirs["joint"] / "eval_results" / "best_acc.pkl"),
                "--bone",
                str(ROOT / "work_dir" / stream_dirs["bone"] / "eval_results" / "best_acc.pkl"),
                "--joint-motion",
                str(ROOT / "work_dir" / stream_dirs["joint_motion"] / "eval_results" / "best_acc.pkl"),
                "--bone-motion",
                str(ROOT / "work_dir" / stream_dirs["bone_motion"] / "eval_results" / "best_acc.pkl"),
                "--out-dir",
                str(variant_out),
            ],
            cwd=ROOT,
        )
        metrics = read_json(variant_out / "metrics.json")
        rows.append(
            {
                "variant": variant,
                "uniform_fusion_top1": round(float(metrics["top1"]) * 100, 2),
                "uniform_fusion_pc": round(float(metrics["top1_per_class"]) * 100, 2),
                "uniform_fusion_top5": round(float(metrics["top5"]) * 100, 2),
                "uniform_fusion_top5_pc": round(float(metrics["top5_per_class"]) * 100, 2),
            }
        )

    csv_path = out_dir / "uniform_fusion_summary.csv"
    write_csv(
        csv_path,
        rows,
        [
            "variant",
            "uniform_fusion_top1",
            "uniform_fusion_pc",
            "uniform_fusion_top5",
            "uniform_fusion_top5_pc",
        ],
    )

    json_path = out_dir / "uniform_fusion_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
