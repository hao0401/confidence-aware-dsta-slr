from __future__ import annotations

from importlib import import_module
from typing import Any


def export_module(module_name: str) -> dict[str, Any]:
    module = import_module(module_name)
    export_names = getattr(module, "__all__", None)
    if export_names is None:
        export_names = [name for name in vars(module) if not name.startswith("_")]
    return {name: getattr(module, name) for name in export_names}


def run_module_main(module_name: str) -> None:
    module = import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"Module {module_name!r} does not define a main() function")
    main()
