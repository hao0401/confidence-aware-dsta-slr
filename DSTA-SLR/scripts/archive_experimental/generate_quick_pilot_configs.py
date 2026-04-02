import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
BASE_CONFIG_DIR = ROOT / "config" / "confidence"
OUTPUT_CONFIG_DIR = ROOT / "config" / "confidence_quick"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")

QUICK_SPECS = {
    "MSASL200": {
        "dataset_key": "msasl200",
        "num_epoch": 10,
        "warm_up_epoch": 2,
        "step": [7],
        "only_train_epoch": 0,
    },
    "MSASL500": {
        "dataset_key": "msasl500",
        "num_epoch": 15,
        "warm_up_epoch": 3,
        "step": [10, 13],
        "only_train_epoch": 0,
    },
    "MSASL1000": {
        "dataset_key": "msasl1000",
        "num_epoch": 20,
        "warm_up_epoch": 3,
        "step": [12, 16],
        "only_train_epoch": 0,
    },
    "NMFs-CSL": {
        "dataset_key": "nmfs_csl",
        "num_epoch": 10,
        "warm_up_epoch": 2,
        "step": [7],
        "only_train_epoch": 0,
    },
}


def replace_scalar(text: str, key: str, value) -> str:
    pattern = rf"^{re.escape(key)}:\s*.*$"
    replacement = f"{key}: {value}"
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0:
        text = text.rstrip() + "\n" + replacement + "\n"
    return text


def replace_list(text: str, key: str, values) -> str:
    pattern = rf"^{re.escape(key)}:\s*\n(?:-\s*.*\n?)+"
    replacement = key + ":\n" + "".join(f"- {value}\n" for value in values)
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0:
        text = text.rstrip() + "\n" + replacement
    return text


def generate_configs(prefix: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, spec in QUICK_SPECS.items():
        dataset_key = spec["dataset_key"]
        for stream_name in STREAMS:
            source_path = BASE_CONFIG_DIR / f"{dataset_key}_{stream_name}.yaml"
            if not source_path.exists():
                raise FileNotFoundError(source_path)

            experiment_name = f"{prefix}_{dataset_key}_{stream_name}"
            text = source_path.read_text(encoding="utf-8")
            text = replace_scalar(text, "Experiment_name", experiment_name)
            text = replace_scalar(text, "wandb_name", experiment_name)
            text = replace_scalar(text, "num_epoch", spec["num_epoch"])
            text = replace_scalar(text, "warm_up_epoch", spec["warm_up_epoch"])
            text = replace_scalar(text, "only_train_epoch", spec["only_train_epoch"])
            text = replace_list(text, "step", spec["step"])

            target_path = output_dir / f"{dataset_key}_{stream_name}.yaml"
            target_path.write_text(text, encoding="utf-8")
            print(target_path)


def main():
    parser = argparse.ArgumentParser(description="Generate short-cycle quick-pilot configs.")
    parser.add_argument("--prefix", default="qp")
    parser.add_argument("--output-dir", default=str(OUTPUT_CONFIG_DIR))
    args = parser.parse_args()

    generate_configs(prefix=args.prefix, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
