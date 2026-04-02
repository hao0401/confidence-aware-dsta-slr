import argparse
import pickle
from pathlib import Path

import numpy as np
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
DATA_ROOT = ROOT / "data"


def load_labels(path):
    with open(path, "rb") as handle:
        names, labels = pickle.load(handle, encoding="latin1")
    return names, labels


def subset_split(data, names, labels, num_classes):
    keep_indices = [idx for idx, label in enumerate(labels) if label < num_classes]
    subset_data = data[keep_indices]
    subset_names = [names[idx] for idx in keep_indices]
    subset_labels = [int(labels[idx]) for idx in keep_indices]
    return subset_data, subset_names, subset_labels


def derive_subset(num_classes):
    source_dir = DATA_ROOT / "WLASL2000"
    target_dir = DATA_ROOT / f"WLASL{num_classes}"
    target_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(source_dir / "train_data_joint.npy")
    train_names, train_labels = load_labels(source_dir / "train_label.pkl")
    subset_train_data, subset_train_names, subset_train_labels = subset_split(
        train_data, train_names, train_labels, num_classes
    )
    np.save(target_dir / "train_data_joint.npy", subset_train_data)
    with open(target_dir / "train_label.pkl", "wb") as handle:
        pickle.dump((subset_train_names, subset_train_labels), handle)

    val_data = np.load(source_dir / "val_data_joint.npy")
    val_names, val_labels = load_labels(source_dir / "val_label.pkl")
    subset_val_data, subset_val_names, subset_val_labels = subset_split(
        val_data, val_names, val_labels, num_classes
    )
    np.save(target_dir / "val_data_joint.npy", subset_val_data)
    with open(target_dir / "val_label.pkl", "wb") as handle:
        pickle.dump((subset_val_names, subset_val_labels), handle)

    print(
        {
            "dataset": f"WLASL{num_classes}",
            "train_samples": len(subset_train_names),
            "val_samples": len(subset_val_names),
            "num_classes": len(set(subset_train_labels + subset_val_labels)),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Derive WLASL300/1000 subsets from the existing WLASL2000 preprocessed data."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[300, 1000],
        choices=[100, 300, 1000],
    )
    args = parser.parse_args()

    for num_classes in args.classes:
        derive_subset(num_classes)


if __name__ == "__main__":
    main()
