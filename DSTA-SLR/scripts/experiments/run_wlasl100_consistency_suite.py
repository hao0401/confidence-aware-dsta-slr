import argparse
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
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_consistency_configs"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")
def config_matches_existing(work_dir: Path, config_path: Path) -> bool:
    copied_config = work_dir / config_path.name
    if not copied_config.exists():
        return True
    return copied_config.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )
def patch_config(
    config,
    experiment_name,
    pred_weight,
    feat_weight,
    noise_std,
    missing_prob,
    use_reliability_weight,
    num_epoch,
):
    config = dict(config)
    config["Experiment_name"] = experiment_name
    config["wandb_name"] = experiment_name
    if num_epoch is not None:
        config["num_epoch"] = num_epoch

    # Keep the legacy field in sync for compatibility/logging.
    config["consistency_loss_weight"] = pred_weight
    config["consistency_pred_loss_weight"] = pred_weight
    config["consistency_feature_loss_weight"] = feat_weight
    config["consistency_use_reliability_weight"] = use_reliability_weight
    config["consistency_noise_std"] = noise_std
    config["consistency_missing_prob"] = missing_prob
    return config


def run_stream(config_path, device, num_worker, overwrite):
    config = read_yaml(config_path)
    experiment_name = config["Experiment_name"]
    work_dir = ROOT / "work_dir" / experiment_name
    save_dir = work_dir / "save_models"
    if work_dir.exists() and not overwrite and not config_matches_existing(work_dir, config_path):
        raise RuntimeError(
            f"Existing work_dir for {experiment_name} was created from a different config. "
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
    if config.get("num_epoch") is not None:
        command.extend(["--num-epoch", str(config["num_epoch"])])
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
    return {
        "work_dir": work_dir,
        "score_path": work_dir / "eval_results" / "best_acc.pkl",
        "model_path": work_dir / "save_models" / "best_model.pt",
        "metrics_path": work_dir / "eval_results" / "best_metrics.json",
    }
def main():
    parser = argparse.ArgumentParser(
        description="Run WLASL100 four-stream consistency training and optional robustness evaluation."
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument("--experiment-prefix", default="conf_wlasl100_consistency")
    parser.add_argument("--pred-weight", type=float, default=0.5)
    parser.add_argument("--feat-weight", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--missing-prob", type=float, default=0.1)
    parser.add_argument(
        "--use-reliability-weight",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--run-robustness", action="store_true")
    args = parser.parse_args()

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fusion_input_dir = ROOT / "work_dir" / f"{args.experiment_prefix}_fusion_inputs"
    fusion_output_dir = ROOT / "work_dir" / f"{args.experiment_prefix}_fusion_results"
    fusion_input_dir.mkdir(parents=True, exist_ok=True)
    fusion_output_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for stream_name in STREAMS:
        base_config_path = CONFIG_DIR / f"wlasl100_{stream_name}.yaml"
        config = read_yaml(base_config_path)
        experiment_name = f"{args.experiment_prefix}_{stream_name}"
        patched = patch_config(
            config=config,
            experiment_name=experiment_name,
            pred_weight=args.pred_weight,
            feat_weight=args.feat_weight,
            noise_std=args.noise_std,
            missing_prob=args.missing_prob,
            use_reliability_weight=args.use_reliability_weight,
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
        shutil.copy2(
            collected[stream_name]["model_path"],
            fusion_input_dir / f"best_model_{stream_name}.pt",
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
            "--out-dir",
            str(fusion_output_dir),
        ],
        cwd=ROOT,
    )

    summary = {
        "experiment_prefix": args.experiment_prefix,
        "pred_weight": args.pred_weight,
        "feat_weight": args.feat_weight,
        "noise_std": args.noise_std,
        "missing_prob": args.missing_prob,
        "streams": {
            stream_name: read_json(info["metrics_path"])
            for stream_name, info in collected.items()
        },
        "fusion": read_json(fusion_output_dir / "metrics.json"),
    }

    if args.run_robustness:
        joint_config_path = TMP_CONFIG_DIR / f"{args.experiment_prefix}_joint.yaml"
        robustness_out = ROOT / "work_dir" / args.experiment_prefix / "robustness.csv"
        run_command(
            [
                PYTHON,
                "scripts/run_robustness_suite.py",
                "--config",
                str(joint_config_path),
                "--weights",
                str(collected["joint"]["model_path"]),
                "--device",
                str(args.device),
                "--num-worker",
                str(args.num_worker),
                "--out-file",
                str(robustness_out),
            ],
            cwd=ROOT,
        )
        summary["robustness_csv"] = str(robustness_out)

    summary_path = ROOT / "work_dir" / f"{args.experiment_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(summary_path)


if __name__ == "__main__":
    main()
