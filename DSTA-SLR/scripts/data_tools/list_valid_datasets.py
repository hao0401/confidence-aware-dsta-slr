import argparse
from pathlib import Path

import numpy as np
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
DATA_ROOT = ROOT / "data"
REQUIRED_FILES = (
    "train_data_joint.npy",
    "train_label.pkl",
    "val_data_joint.npy",
    "val_label.pkl",
)


def dataset_is_valid(dataset_dir: Path) -> bool:
    for name in REQUIRED_FILES:
        path = dataset_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return False
    try:
        train_data = np.load(dataset_dir / "train_data_joint.npy", mmap_mode="r")
        val_data = np.load(dataset_dir / "val_data_joint.npy", mmap_mode="r")
    except Exception:  # noqa: BLE001
        return False
    return (
        len(train_data.shape) == 5
        and len(val_data.shape) == 5
        and train_data.shape[1] == 3
        and val_data.shape[1] == 3
    )


def main():
    parser = argparse.ArgumentParser(description="List locally valid benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset names.",
    )
    args = parser.parse_args()

    if args.datasets:
        candidates = [DATA_ROOT / name for name in args.datasets]
    else:
        candidates = [path for path in DATA_ROOT.iterdir() if path.is_dir()]

    for dataset_dir in candidates:
        if dataset_is_valid(dataset_dir):
            print(dataset_dir.name)


if __name__ == "__main__":
    main()
