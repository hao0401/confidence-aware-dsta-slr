from _compat import export_module as _export_module, run_module_main as _run_module_main

_MODULE = "experiments.run_wlasl100_joint_repeats_minimal"
globals().update(_export_module(_MODULE))
__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    _run_module_main(_MODULE)
