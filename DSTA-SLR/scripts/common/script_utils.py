from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


def find_repo_root(current_file: str | Path) -> Path:
    current_path = Path(current_file).resolve()
    for candidate in (current_path.parent, *current_path.parents):
        if (candidate / "main.py").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {current_path}")


def run_command(command, cwd: Path, flush: bool = False) -> None:
    print(" ".join(str(part) for part in command), flush=flush)
    subprocess.run(command, cwd=cwd, check=True)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, content) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_checkpoint(save_dir: Path):
    candidates = []
    for path in save_dir.glob("epoch-*.pt"):
        try:
            epoch = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        candidates.append((epoch, path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])
