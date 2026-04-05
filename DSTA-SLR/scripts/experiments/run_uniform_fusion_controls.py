import argparse
from pathlib import Path

from script_utils import (
    experiment_artifact_paths,
    extract_metric_fields,
    find_repo_root,
    read_json,
    run_fusion,
    UNIFORM_FUSION_SUMMARY_FIELDS,
    write_csv,
    write_json,
)
from python_locator import resolve_python


ROOT = find_repo_root(__file__)
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

    label_path = Path(args.label_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in args.variants:
        stream_dirs = VARIANT_STREAM_DIRS[variant]
        variant_out = out_dir / variant
        run_fusion(
            ROOT,
            PYTHON,
            script_path="ensemble/fuse_streams_uniform.py",
            label_path=label_path,
            score_paths={
                stream_name: experiment_artifact_paths(ROOT, stream_dirs[stream_name])["score_path"]
                for stream_name in ("joint", "bone", "joint_motion", "bone_motion")
            },
            out_dir=variant_out,
        )
        metrics = read_json(variant_out / "metrics.json")
        rows.append(
            {
                "variant": variant,
                **extract_metric_fields(
                    metrics,
                    field_map={
                        "top1": "uniform_fusion_top1",
                        "top1_per_class": "uniform_fusion_pc",
                        "top5": "uniform_fusion_top5",
                        "top5_per_class": "uniform_fusion_top5_pc",
                    },
                    scale=100.0,
                    round_digits=2,
                ),
            }
        )

    csv_path = out_dir / "uniform_fusion_summary.csv"
    write_csv(
        csv_path,
        rows,
        UNIFORM_FUSION_SUMMARY_FIELDS,
    )

    json_path = out_dir / "uniform_fusion_summary.json"
    write_json(json_path, rows)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
