"""Canonical performance terminology and read-only legacy-data adapters.

PORTAL characterizes optimizer behavior with checkpoint errors and aggregated
performance vectors. Historical ISA-named files/keys are accepted only as
compatibility inputs; these adapters do not implement Instance Space Analysis.
Existing measurement files are never renamed or rewritten here.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OLD = re.compile(r"(?<![A-Za-z])isa(?![A-Za-z])", re.IGNORECASE)
_NEW = re.compile(r"(?<![A-Za-z])performance(?![A-Za-z])", re.IGNORECASE)


def performance_name(text: str) -> str:
    """Translate old identifier components without changing words like 'isalpha'."""
    def replace(match):
        word = match.group(0)
        replacement = "PERFORMANCE" if word.isupper() else "performance"
        return _OLD.sub(replacement, word)
    return re.sub(r"\bPERFORMANCE\b", "Performance", _WORDS.sub(replace, str(text)))


def normalize_performance_keys(value: Any) -> Any:
    """Read legacy configuration/metadata keys without touching path strings."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            name = performance_name(key) if isinstance(key, str) else key
            item = normalize_performance_keys(item)
            if name in output and output[name] != item:
                raise ValueError(f"Conflicting legacy/canonical performance key: {name}")
            output[name] = item
        return output
    if isinstance(value, list):
        return [normalize_performance_keys(item) for item in value]
    if isinstance(value, str) and value in ("isa", "ISA"):
        return "performance" if value == "isa" else "Performance"
    return value


def resolve_performance_input(path: str | Path) -> Path:
    """Prefer the requested input, falling back to its historical ISA filename."""
    requested = Path(path)
    if requested.exists():
        return requested
    def legacy_name(name, case=None):
        def replace(match):
            word = match.group(0)
            replacement = case or ("ISA" if word.isupper() or word == "Performance" else "isa")
            return _NEW.sub(replacement, word)
        return _WORDS.sub(replace, name)
    # Try preserved case first, then the two historical spellings. The latter
    # also support old mixed-case directory names on case-sensitive systems.
    for legacy in dict.fromkeys(Path(legacy_name(str(requested), case))
                                for case in (None, "ISA", "isa")):
        if legacy != requested and legacy.exists():
            return legacy
    # Directory and filename capitalization may differ in older study outputs.
    candidate = Path(requested.anchor)
    parts = requested.parts[1:] if requested.anchor else requested.parts
    for part in parts:
        names = dict.fromkeys([part, *(legacy_name(part, case) for case in (None, "ISA", "isa"))])
        options = [candidate / name for name in names]
        candidate = next((option for option in options if option.exists()), options[0])
    if candidate.exists():
        return candidate
    return requested


def read_characterization_csv(source, *args, **kwargs):
    """Read canonical or legacy measurement tables using canonical field names."""
    import pandas as pd
    is_url = isinstance(source, str) and "://" in source
    path = resolve_performance_input(source) if isinstance(source, (str, Path)) and not is_url else source
    frame = pd.read_csv(path, *args, **kwargs)
    names = [performance_name(column) if isinstance(column, str) else column
             for column in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError("Legacy and canonical performance columns would collide.")
    frame.columns = names
    for column in ("space", "representation", "rule", "feature_space"):
        if column in frame:
            frame[column] = frame[column].replace({"isa": "performance", "ISA": "Performance"})
    return frame


def load_characterization_json(source, *args, **kwargs):
    return normalize_performance_keys(json.load(source, *args, **kwargs))


def loads_characterization_json(source, *args, **kwargs):
    return normalize_performance_keys(json.loads(source, *args, **kwargs))
