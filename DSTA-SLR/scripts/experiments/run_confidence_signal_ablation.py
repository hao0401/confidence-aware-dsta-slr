import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from python_locator import resolve_python
from script_utils import (
    find_latest_checkpoint,
    find_repo_root,
    read_json,
    read_yaml,
    run_command,
    write_csv,
    write_yaml,
)

ROOT = find_repo_root(__file__)

PYTHON = resolve_python(ROOT)
if (
    Path(sys.executable).resolve() != Path(PYTHON).resolve()
    and os.environ.get("DSTA_SLR_SKIP_REEXEC") != "1"
):
    env = os.environ.copy()
    env["DSTA_SLR_SKIP_REEXEC"] = "1"
    subprocess.run(
        [PYTHON, str(ROOT / "scripts" / Path(__file__).name), *sys.argv[1:]],
        cwd=ROOT,
        check=True,
        env=env,
    )
    sys.exit(0)


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
    work_dir = ROOT / "work_dir" / experiment_name
    save_dir = work_dir / "save_models"
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
    command = [
        PYTHON,
        "-u",
        "main.py",
        "--config",
        str(config_path),
        "--device",
        str(device),
        "--num-worker",
        str(num_worker),
    ]
    if overwrite:
        command.extend(["--overwrite-work-dir", "true"])
    elif save_dir.exists():
        latest_epoch, latest_ckpt = find_latest_checkpoint(save_dir)
        if latest_ckpt is not None:
            command.extend(
                [
                    "--weights",
                    str(latest_ckpt),
                    "--start-epoch",
                    str(latest_epoch + 1),
                ]
            )
    run_command(command, cwd=ROOT)
    fingerprint_path(work_dir).write_text(
        compute_code_fingerprint(), encoding="utf-8"
    )
    return {
        "work_dir": work_dir,
        "score_path": work_dir / "eval_results" / "best_acc.pkl",
        "model_path": work_dir / "save_models" / "best_model.pt",
        "metrics_path": work_dir / "eval_results" / "best_metrics.json",
    }
def main():
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
    args = parser.parse_args()

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for variant_name in args.variants:
        variant_spec = VARIANTS[variant_name]
        fusion_input_dir = (
            ROOT / "work_dir" / f"{args.experiment_prefix}_{variant_name}_fusion_inputs"
        )
        fusion_output_dir = (
            ROOT / "work_dir" / f"{args.experiment_prefix}_{variant_name}_fusion_results"
        )
        fusion_input_dir.mkdir(parents=True, exist_ok=True)
        fusion_output_dir.mkdir(parents=True, exist_ok=True)
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
                fusion_input_dir / f"best_acc_{stream_name}.pkl",
            )

        run_command(
            [
                PYTHON,
                "ensemble/fuse_streams.py",
                "--label-path",
                str(ROOT / "data" / "WLASL100" / "val_label.pkl"),
                "--data-path",
                str(ROOT / "data" / "WLASL100" / "val_data_joint.npy"),
                "--joint",
                str(fusion_input_dir / "best_acc_joint.pkl"),
                "--bone",
                str(fusion_input_dir / "best_acc_bone.pkl"),
                "--joint-motion",
                str(fusion_input_dir / "best_acc_joint_motion.pkl"),
                "--bone-motion",
                str(fusion_input_dir / "best_acc_bone_motion.pkl"),
                "--window-size",
                "120",
                "--confidence-mode",
                variant_spec["confidence_mode"],
                "--confidence-constant-value",
                str(variant_spec["confidence_constant_value"]),
                "--out-dir",
                str(fusion_output_dir),
            ],
            cwd=ROOT,
        )

        fusion_metrics = read_json(fusion_output_dir / "metrics.json")
        joint_metrics = read_json(collected["joint"]["metrics_path"])
        summary_rows.append(
            {
                "variant": variant_name,
                "joint_top1": joint_metrics["top1"],
                "joint_top1_per_class": joint_metrics["top1_per_class"],
                "fusion_top1": fusion_metrics["top1"],
                "fusion_top1_per_class": fusion_metrics["top1_per_class"],
            }
        )

    summary_dir = ROOT / "work_dir" / args.experiment_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        raise RuntimeError("No variants were executed; summary_rows is empty.")
    csv_path = summary_dir / "confidence_signal_ablation.csv"
    write_csv(csv_path, summary_rows, list(summary_rows[0].keys()))

    json_path = summary_dir / "confidence_signal_ablation.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
