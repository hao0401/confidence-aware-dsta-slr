from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _normalize_sys_path_entry(entry: str) -> str:
    return str(Path(entry or ".").resolve())


def ensure_sys_path(path: str | Path) -> None:
    target = str(Path(path).resolve())
    if target not in {_normalize_sys_path_entry(entry) for entry in sys.path}:
        sys.path.insert(0, target)


def maybe_reexec_with_python(
    root: str | Path,
    current_file: str | Path,
    python_executable: str | Path,
    argv: list[str] | None = None,
    *,
    skip_env_var: str = "DSTA_SLR_SKIP_REEXEC",
) -> bool:
    root_path = Path(root).resolve()
    target_python = Path(python_executable).resolve()
    if Path(sys.executable).resolve() == target_python:
        return False
    if os.environ.get(skip_env_var) == "1":
        return False

    wrapper_path = root_path / "scripts" / Path(current_file).name
    if not wrapper_path.exists():
        raise FileNotFoundError(f"Could not find wrapper script for {current_file}: {wrapper_path}")

    env = os.environ.copy()
    env[skip_env_var] = "1"
    subprocess.run(
        [str(target_python), str(wrapper_path), *(sys.argv[1:] if argv is None else argv)],
        cwd=root_path,
        check=True,
        env=env,
    )
    return True


__all__ = ["ensure_sys_path", "maybe_reexec_with_python"]
