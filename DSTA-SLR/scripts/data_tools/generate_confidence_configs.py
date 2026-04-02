from pathlib import Path

from experiment_specs import DATASET_SPECS, STREAM_SPECS
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
CONFIG_DIR = ROOT / "config" / "confidence"


def build_config(dataset_name, stream_name):
    dataset_spec = DATASET_SPECS[dataset_name]
    stream_spec = STREAM_SPECS[stream_name]
    experiment_name = f"conf_{dataset_name.lower().replace('-', '_')}_{stream_name}"
    common_feeder = {
        "debug": False,
        "window_size": dataset_spec["window_size"],
        "normalization": True,
        "lap_pe": False,
        "is_vector": False,
        "bone_stream": stream_spec["bone_stream"],
        "motion_stream": stream_spec["motion_stream"],
        "missing_joint_prob": 0.0,
        "noise_std": 0.0,
    }
    return {
        "Experiment_name": experiment_name,
        "dataset": dataset_name,
        "feeder": "feeders.feeder.Feeder",
        "train_feeder_args": {
            **common_feeder,
            "random_choose": True,
            "random_shift": True,
            "random_mirror": True,
            "random_mirror_p": 0.5,
        },
        "test_feeder_args": {
            **common_feeder,
            "random_choose": False,
            "random_mirror": False,
        },
        "model": "model.fstgan.Model",
        "model_args": {
            "num_class": dataset_spec["num_class"],
            "num_point": 27,
            "num_person": 1,
            "graph": "graph.sign_27.Graph",
            "groups": 16,
            "block_size": 41,
            "graph_args": {"labeling_mode": "spatial"},
            "inner_dim": 64,
            "depth": 4,
            "drop_layers": 2,
            "use_confidence_encoding": True,
            "use_confidence_graph": True,
            "confidence_graph_lambda": 1.0,
            "use_temporal_rectification": True,
            "temporal_window_size": 5,
        },
        "weight_decay": 0.0001,
        "base_lr": 0.1,
        "step": [150, 200],
        "device": [0],
        "keep_rate": 0.9,
        "only_train_epoch": 1,
        "batch_size": dataset_spec["batch_size"],
        "test_batch_size": dataset_spec["batch_size"],
        "num_epoch": 250,
        "nesterov": True,
        "warm_up_epoch": 20,
        "wandb": False,
        "wandb_project": "DSTA-SLR Confidence Robustness",
        "wandb_entity": "irvl",
        "wandb_name": experiment_name,
        "num_worker": 4,
        "save_interval": 5,
        "overwrite_work_dir": False,
        "consistency_loss_weight": 0.1,
        "consistency_noise_std": 0.01,
        "consistency_missing_prob": 0.1,
    }


def main():
    import yaml

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASET_SPECS:
        for stream_name in STREAM_SPECS:
            config = build_config(dataset_name, stream_name)
            output_path = CONFIG_DIR / f"{dataset_name.lower().replace('-', '_')}_{stream_name}.yaml"
            with open(output_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            print(output_path)


if __name__ == "__main__":
    main()
