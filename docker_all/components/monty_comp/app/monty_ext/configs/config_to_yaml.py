"""Convert experiment config dict (with class/instance refs) to YAML-serializable form.

Used to generate conf/*.yaml documentation from build_config() so repetitive
structures (e.g. 100 sensor modules, sm_to_lm_matrix) are emitted in full
without hand-writing every entry.
"""

from __future__ import annotations

import types
from typing import Any


def _class_path(obj: type | types.FunctionType) -> str:
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__name__', repr(obj))}"


def config_to_yaml_safe(obj: Any) -> Any:
    """Recursively convert config values to types safe for PyYAML dump.

    - type/class -> "module.ClassName"
    - function -> "module.func_name"
    - instance with __dict__ -> {"class": "module.ClassName", **serialized __dict__}
    - tuple -> list
    - numpy array -> list (via .tolist())
    - dict/list -> recurse
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, type):
        return _class_path(obj)
    if isinstance(obj, types.FunctionType):
        return _class_path(obj)
    if isinstance(obj, (list, tuple)):
        return [config_to_yaml_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: config_to_yaml_safe(v) for k, v in obj.items()}
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    # Instance: serialize as class path + __dict__
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        out = {"class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"}
        for k, v in getattr(obj, "__dict__", {}).items():
            if k.startswith("_"):
                continue
            try:
                out[k] = config_to_yaml_safe(v)
            except Exception:
                out[k] = repr(v)
        return out
    return obj
