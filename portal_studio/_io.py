"""I/O helpers for datasets and configurations.

Supports reading and writing small numeric datasets used by the compute CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    import numpy as np


def load_config_json(path: str | Path) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    TODO: Validate the instance schema if this loader is used for instance files.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config_json(path: str | Path, cfg: Dict[str, Any]) -> None:
    """Save configuration to a JSON file."""

    p = Path(path)
    def _sanitize(obj: Any) -> Any:
        """Recursively convert non-JSON-native objects (e.g., numpy arrays/scalars)
        into JSON-serializable Python types.
        """
        # Local import to avoid hard dependency at module import time
        try:
            import numpy as _np
        except Exception:
            _np = None

        if _np is not None:
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.floating, _np.integer)):
                return obj.item()
        # Built-in JSON types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        # Fallback: try to use `tolist` if available, else convert to string
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        return str(obj)

    safe_cfg = _sanitize(cfg)
    with p.open("w", encoding="utf-8") as f:
        json.dump(safe_cfg, f, indent=2, sort_keys=True)


def read_array(path: str | Path, fmt: str | None = None) -> "np.ndarray":
    """Read an array from CSV, NPY, NPZ, or JSON.

    - CSV: headerless numeric, returns 2D array
    - NPY: np.load
    - NPZ: expects key 'X' by convention; falls back to first array
    - JSON: top-level list (or list of lists)
    """

    import numpy as np

    p = Path(path)
    suffix = p.suffix.lower()
    ext = fmt.lower() if fmt else suffix.lstrip(".")

    if ext in {"csv"}:
        arr = np.genfromtxt(p, delimiter=",", dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    if ext in {"npy"}:
        return np.load(p)
    if ext in {"npz"}:
        data = np.load(p)
        if "X" not in data.files:
            raise ValueError("NPZ input must contain key 'X' for dataset array")
        return data["X"]
    if ext in {"json"}:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        # Expect top-level list or list of lists of numbers
        if not isinstance(obj, (list, tuple)):
            raise ValueError("JSON input must be a list or list of lists of numbers")
        try:
            arr = np.array(obj, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"JSON input contains non-numeric values: {e}")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    raise ValueError(f"Unsupported input format: {ext}")


def write_array(path: str | Path, arr: "np.ndarray", fmt: str | None = None) -> None:
    """Write an array to CSV, NPY, NPZ, or JSON.

    - CSV: writes 2D array with commas
    - NPY: saves as .npy
    - NPZ: saves with key 'y'
    - JSON: writes list or list of lists
    """

    import numpy as np

    p = Path(path)
    suffix = p.suffix.lower()
    ext = fmt.lower() if fmt else suffix.lstrip(".")
    arr = np.asarray(arr, dtype=np.float64)

    if ext in {"csv"}:
        np.savetxt(p, arr, delimiter=",", fmt="%.17g")
        return
    if ext in {"npy"}:
        np.save(p, arr)
        return
    if ext in {"npz"}:
        np.savez(p, y=arr)
        return
    if ext in {"json"}:
        with p.open("w", encoding="utf-8") as f:
            json.dump(arr.tolist(), f)
        return
    raise ValueError(f"Unsupported output format: {ext}")
