import argparse
import shutil
from pathlib import Path

from experiment_specs import STREAM_SPECS
from python_locator import resolve_python
from script_utils import find_latest_checkpoint, find_repo_root, run_command


ROOT = find_repo_root(__file__)
CONFIG_DIR = ROOT / "config" / "confidence"
PYTHON = resolve_python(ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run confidence-aware four-stream training and fusion.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Directory containing per-stream YAML configs.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="conf",
        help="Prefix used for work_dir experiment names and fusion result directories.",
    )
    args = parser.parse_args()

    dataset_key = args.dataset.lower().replace("-", "_")
    config_dir = Path(args.config_dir)
    fusion_base = f"{args.experiment_prefix}_{dataset_key}" if args.experiment_prefix else dataset_key
    fusion_input_dir = ROOT / "work_dir" / f"{fusion_base}_fusion_inputs"
    fusion_output_dir = ROOT / "work_dir" / f"{fusion_base}_fusion_results"
    fusion_input_dir.mkdir(parents=True, exist_ok=True)
    fusion_output_dir.mkdir(parents=True, exist_ok=True)

    for stream_name in STREAM_SPECS:
        config_path = config_dir / f"{dataset_key}_{stream_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        experiment_name = (
            f"{args.experiment_prefix}_{dataset_key}_{stream_name}"
            if args.experiment_prefix
            else f"{dataset_key}_{stream_name}"
        )
        source_dir = ROOT / "work_dir" / experiment_name
        save_dir = source_dir / "save_models"
        command = [
            PYTHON,
            "-u",
            "main.py",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--num-worker",
            str(args.num_worker),
        ]
        if args.overwrite_work_dir:
            command.extend(["--overwrite-work-dir", "true"])
        if args.num_epoch is not None:
            command.extend(["--num-epoch", str(args.num_epoch)])
        if not args.overwrite_work_dir and save_dir.exists():
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
        shutil.copy2(
            source_dir / "eval_results" / "best_acc.pkl",
            fusion_input_dir / f"best_acc_{stream_name}.pkl",
        )
        shutil.copy2(
            source_dir / "save_models" / "best_model.pt",
            fusion_input_dir / f"best_model_{stream_name}.pt",
        )

    run_command(
        [
            PYTHON,
            "ensemble/fuse_streams.py",
            "--label-path",
            str(ROOT / "data" / args.dataset / "val_label.pkl"),
            "--data-path",
            str(ROOT / "data" / args.dataset / "val_data_joint.npy"),
            "--joint",
            str(fusion_input_dir / "best_acc_joint.pkl"),
            "--bone",
            str(fusion_input_dir / "best_acc_bone.pkl"),
            "--joint-motion",
            str(fusion_input_dir / "best_acc_joint_motion.pkl"),
            "--bone-motion",
            str(fusion_input_dir / "best_acc_bone_motion.pkl"),
            "--out-dir",
            str(fusion_output_dir),
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
