from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


ARCHIVE_DIR = Path(__file__).resolve().parent
COMMON_DIR = ARCHIVE_DIR.parent / "common"


@lru_cache(maxsize=None)
def load_common_module(module_name: str) -> ModuleType:
    module_path = COMMON_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing common module: {module_path}")

    qualified_name = f"_archive_common_{module_name}"
    existing_module = sys.modules.get(qualified_name)
    if existing_module is not None:
        return existing_module

    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_common_module"]
