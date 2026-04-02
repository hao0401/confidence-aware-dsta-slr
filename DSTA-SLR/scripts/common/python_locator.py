import os
import sys
from pathlib import Path


def resolve_python(root: Path) -> str:
    conda_env_python = Path(r"C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe")
    if conda_env_python.exists():
        return str(conda_env_python)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "python.exe"
        if candidate.exists():
            return str(candidate)

    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    return str(Path(sys.executable))
