import os
import subprocess
import sys
from pathlib import Path
from shutil import which


def _resolve_existing_path(path_like: str | Path | None) -> str | None:
    if not path_like:
        return None
    candidate = Path(path_like)
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _conda_env_python() -> str | None:
    conda_exe = (
        _resolve_existing_path(os.environ.get("DSTA_SLR_CONDA_EXE"))
        or _resolve_existing_path(os.environ.get("CONDA_EXE"))
        or which("conda.exe")
        or which("conda")
        or which("conda.bat")
    )
    if not conda_exe:
        return None

    try:
        conda_base = subprocess.check_output(
            [conda_exe, "info", "--base"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    return _resolve_existing_path(Path(conda_base) / "envs" / "dsta-slr" / "python.exe")


def resolve_python(root: Path) -> str:
    explicit_python = _resolve_existing_path(os.environ.get("DSTA_SLR_PYTHON"))
    if explicit_python:
        return explicit_python

    conda_env_python = _conda_env_python()
    if conda_env_python:
        return conda_env_python

    for relative_path in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        candidate = _resolve_existing_path(root.joinpath(*relative_path))
        if candidate:
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for relative_path in ("python.exe", "bin/python"):
            candidate = _resolve_existing_path(Path(conda_prefix) / relative_path)
            if candidate:
                return candidate

    return str(Path(sys.executable))


__all__ = ["resolve_python"]
