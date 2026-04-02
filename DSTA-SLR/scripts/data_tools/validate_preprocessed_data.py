import argparse
from pathlib import Path

import numpy as np
import pickle

from experiment_specs import DATASET_SPECS
from script_utils import find_repo_root

ROOT = find_repo_root(__file__)


REQUIRED_FILES = (
    "train_data_joint.npy",
    "train_label.pkl",
    "val_data_joint.npy",
    "val_label.pkl",
)


def validate_dataset(root, dataset_name):
    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        return {"dataset": dataset_name, "status": "missing_dir"}

    missing_files = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing_files:
        return {
            "dataset": dataset_name,
            "status": "missing_files",
            "missing_files": missing_files,
        }

    train_data = np.load(dataset_dir / "train_data_joint.npy", mmap_mode="r")
    val_data = np.load(dataset_dir / "val_data_joint.npy", mmap_mode="r")
    with open(dataset_dir / "train_label.pkl", "rb") as handle:
        train_names, train_labels = pickle.load(handle, encoding="latin1")
    with open(dataset_dir / "val_label.pkl", "rb") as handle:
        val_names, val_labels = pickle.load(handle, encoding="latin1")
    checks = {
        "train_shape": tuple(train_data.shape),
        "val_shape": tuple(val_data.shape),
        "train_label_count": len(train_labels),
        "train_name_count": len(train_names),
        "val_label_count": len(val_labels),
        "val_name_count": len(val_names),
        "train_confidence_min": float(train_data[:, 2].min()),
        "train_confidence_max": float(train_data[:, 2].max()),
        "val_confidence_min": float(val_data[:, 2].min()),
        "val_confidence_max": float(val_data[:, 2].max()),
    }
    shape_ok = (
        len(train_data.shape) == 5
        and len(val_data.shape) == 5
        and train_data.shape[1] == 3
        and val_data.shape[1] == 3
        and train_data.shape[3] == 27
        and val_data.shape[3] == 27
    )
    conf_ok = (
        checks["train_confidence_min"] >= 0.0
        and checks["train_confidence_max"] <= 1.25
        and checks["val_confidence_min"] >= 0.0
        and checks["val_confidence_max"] <= 1.25
    )
    count_ok = (
        train_data.shape[0] == len(train_labels) == len(train_names)
        and val_data.shape[0] == len(val_labels) == len(val_names)
    )
    if shape_ok and conf_ok and count_ok:
        checks["status"] = "ok"
    elif shape_ok and conf_ok:
        checks["status"] = "warning_count_mismatch"
    elif shape_ok:
        checks["status"] = "warning_confidence_range"
    else:
        checks["status"] = "invalid"
    return {"dataset": dataset_name, **checks}


def main():
    parser = argparse.ArgumentParser(description="Validate DSTA-SLR preprocessed data.")
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
        help="Root folder containing dataset subdirectories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_SPECS.keys()),
        help="Datasets to validate.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    for dataset_name in args.datasets:
        result = validate_dataset(data_root, dataset_name)
        print(result)


if __name__ == "__main__":
    main()
