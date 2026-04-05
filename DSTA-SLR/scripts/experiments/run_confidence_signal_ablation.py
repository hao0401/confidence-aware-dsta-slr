import argparse
import hashlib
import shutil
from pathlib import Path

from common.runtime_helpers import maybe_reexec_with_python
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    CONFIDENCE_SIGNAL_ABLATION_FIELDS,
    experiment_artifact_paths,
    extract_metric_fields,
    find_repo_root,
    has_best_artifacts,
    load_best_artifacts,
    maybe_append_resume_args,
    prepare_fusion_workspace,
    read_json,
    read_yaml,
    run_command,
    run_fusion,
    write_csv,
    write_json,
    write_yaml,
)

ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)

CONFIG_DIR = ROOT / "config" / "confidence"
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_conf_signal_configs"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
FINGERPRINT_FILES = (
    ROOT / "main.py",
    ROOT / "model" / "fstgan.py",
    ROOT / "feeders" / "feeder.py",
)

VARIANTS = {
    "original_confidence": {
        "confidence_mode": "original",
        "confidence_constant_value": 1.0,
        "drop_confidence_channel": False,
        "model_args": {
            "use_confidence_encoding": True,
            "use_confidence_graph": True,
            "use_temporal_rectification": True,
            "has_confidence_input": True,
        },
        "consistency_pred_loss_weight": 0.0,
        "consistency_feature_loss_weight": 0.0,
    },
    "constant_confidence": {
        "confidence_mode": "constant",
        "confidence_constant_value": 1.0,
        "drop_confidence_channel": False,
        "model_args": {
            "use_confidence_encoding": True,
            "use_confidence_graph": True,
            "use_temporal_rectification": True,
            "has_confidence_input": True,
        },
        "consistency_pred_loss_weight": 0.0,
        "consistency_feature_loss_weight": 0.0,
    },
    "shuffled_confidence": {
        "confidence_mode": "shuffle",
        "confidence_constant_value": 1.0,
        "drop_confidence_channel": False,
        "model_args": {
            "use_confidence_encoding": True,
            "use_confidence_graph": True,
            "use_temporal_rectification": True,
            "has_confidence_input": True,
        },
        "consistency_pred_loss_weight": 0.0,
        "consistency_feature_loss_weight": 0.0,
    },
    "no_confidence_signal": {
        "confidence_mode": "constant",
        "confidence_constant_value": 1.0,
        "drop_confidence_channel": True,
        "model_args": {
            "in_channels": 2,
            "has_confidence_input": False,
            "use_confidence_encoding": False,
            "use_confidence_graph": False,
            "use_temporal_rectification": False,
        },
        "consistency_pred_loss_weight": 0.0,
        "consistency_feature_loss_weight": 0.0,
    },
}


def compute_code_fingerprint():
    digest = hashlib.sha256()
    for path in FINGERPRINT_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint_path(work_dir: Path) -> Path:
    return work_dir / "code_fingerprint.txt"


def config_matches_existing(work_dir: Path, config_path: Path) -> bool:
    copied_config = work_dir / config_path.name
    if not copied_config.exists():
        return True
    return copied_config.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )


def fingerprint_matches_existing(work_dir: Path) -> bool:
    path = fingerprint_path(work_dir)
    if not path.exists():
        return True
    return path.read_text(encoding="utf-8").strip() == compute_code_fingerprint()


def patch_config(config, experiment_name, variant_spec, num_epoch):
    config = dict(config)
    config["Experiment_name"] = experiment_name
    config["wandb_name"] = experiment_name
    if num_epoch is not None:
        config["num_epoch"] = num_epoch

    config["train_feeder_args"] = dict(config["train_feeder_args"])
    config["test_feeder_args"] = dict(config["test_feeder_args"])
    config["model_args"] = dict(config["model_args"])

    for split_key in ("train_feeder_args", "test_feeder_args"):
        config[split_key]["confidence_mode"] = variant_spec["confidence_mode"]
        config[split_key]["confidence_constant_value"] = variant_spec[
            "confidence_constant_value"
        ]
        config[split_key]["drop_confidence_channel"] = variant_spec.get(
            "drop_confidence_channel", False
        )

    config["model_args"].update(variant_spec["model_args"])
    config["consistency_loss_weight"] = variant_spec["consistency_pred_loss_weight"]
    config["consistency_pred_loss_weight"] = variant_spec[
        "consistency_pred_loss_weight"
    ]
    config["consistency_feature_loss_weight"] = variant_spec[
        "consistency_feature_loss_weight"
    ]
    return config


def run_stream(config_path, device, num_worker, overwrite):
    config = read_yaml(config_path)
    experiment_name = config["Experiment_name"]
    artifacts = experiment_artifact_paths(ROOT, experiment_name)
    work_dir = artifacts["work_dir"]
    save_dir = artifacts["save_dir"]
    if work_dir.exists() and not overwrite:
        if not config_matches_existing(work_dir, config_path):
            raise RuntimeError(
                f"Existing work_dir for {experiment_name} was created from a different config. "
                "Use --overwrite-work-dir or change --experiment-prefix."
            )
        if not fingerprint_matches_existing(work_dir):
            raise RuntimeError(
                f"Existing work_dir for {experiment_name} was created from different code. "
                "Use --overwrite-work-dir or change --experiment-prefix."
            )
        if has_best_artifacts(artifacts):
            return load_best_artifacts(ROOT, experiment_name)
    command = build_main_command(
        PYTHON,
        config_path=config_path,
        device=device,
        num_worker=num_worker,
        overwrite_work_dir=overwrite,
    )
    if not overwrite:
        maybe_append_resume_args(command, save_dir)
    run_command(command, cwd=ROOT)
    fingerprint_path(work_dir).write_text(
        compute_code_fingerprint(), encoding="utf-8"
    )
    return load_best_artifacts(ROOT, experiment_name)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run confidence signal validity ablations on WLASL100."
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=tuple(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
    )
    parser.add_argument("--experiment-prefix", default="conf_signal_wlasl100")
    return parser


def main(argv=None):
    if argv is None and maybe_reexec_with_python(ROOT, __file__, PYTHON):
        return

    args = build_parser().parse_args(argv)

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for variant_name in args.variants:
        variant_spec = VARIANTS[variant_name]
        fusion_workspace = prepare_fusion_workspace(
            ROOT,
            f"{args.experiment_prefix}_{variant_name}",
            stream_names=STREAMS,
        )
        fusion_output_dir = fusion_workspace["output_dir"]
        staged_score_paths = fusion_workspace["score_paths"]
        collected = {}

        for stream_name in STREAMS:
            base_config_path = CONFIG_DIR / f"wlasl100_{stream_name}.yaml"
            config = read_yaml(base_config_path)
            experiment_name = f"{args.experiment_prefix}_{variant_name}_{stream_name}"
            patched = patch_config(
                config=config,
                experiment_name=experiment_name,
                variant_spec=variant_spec,
                num_epoch=args.num_epoch,
            )
            config_path = TMP_CONFIG_DIR / f"{experiment_name}.yaml"
            write_yaml(config_path, patched)
            collected[stream_name] = run_stream(
                config_path=config_path,
                device=args.device,
                num_worker=args.num_worker,
                overwrite=args.overwrite_work_dir,
            )
            shutil.copy2(
                collected[stream_name]["score_path"],
                staged_score_paths[stream_name],
            )

        run_fusion(
            ROOT,
            PYTHON,
            label_path=ROOT / "data" / "WLASL100" / "val_label.pkl",
            data_path=ROOT / "data" / "WLASL100" / "val_data_joint.npy",
            score_paths=staged_score_paths,
            out_dir=fusion_output_dir,
            window_size=120,
            extra_args=[
                "--confidence-mode",
                variant_spec["confidence_mode"],
                "--confidence-constant-value",
                str(variant_spec["confidence_constant_value"]),
            ],
        )

        fusion_metrics = read_json(fusion_output_dir / "metrics.json")
        joint_metrics = collected["joint"]["metrics"]
        summary_rows.append(
            {
                "variant": variant_name,
                **extract_metric_fields(
                    joint_metrics,
                    fields=("top1", "top1_per_class"),
                    prefix="joint_",
                ),
                **extract_metric_fields(
                    fusion_metrics,
                    fields=("top1", "top1_per_class"),
                    prefix="fusion_",
                ),
            }
        )

    summary_dir = ROOT / "work_dir" / args.experiment_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        raise RuntimeError("No variants were executed; summary_rows is empty.")
    csv_path = summary_dir / "confidence_signal_ablation.csv"
    write_csv(csv_path, summary_rows, CONFIDENCE_SIGNAL_ABLATION_FIELDS)

    json_path = summary_dir / "confidence_signal_ablation.json"
    write_json(json_path, summary_rows)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
