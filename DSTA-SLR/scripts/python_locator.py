from _compat import export_module as _export_module

_MODULE = "common.python_locator"
globals().update(_export_module(_MODULE))
__all__ = [name for name in globals() if not name.startswith("_")]
