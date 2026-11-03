"""Configuration loading and validation for Type 3 dataset augmentation."""

from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "portal_type3_dataset",
        "sources": ["Benchmark_S50"],
        "output_root": "augmentation_runs/portal_type3_dataset",
        "resume": True,
    },
    "generation": {
        "count_per_recipe": 100,
        "start_sample": 1,
        "base_seed": 42,
        "center_min": -80.0,
        "center_max": 80.0,
        "beta_min": -1000.0,
        "beta_max": 1000.0,
        "beta_gap": 20.0,
        "beta_opt_gap": 1.0,
        "ensure_no_cover": True,
        "no_cover_max_attempts": 1000,
        "no_cover_tolerance": 1e-6,
        "trace_metadata": False,
    },
    "execution": {
        "max_workers": 0,
        "cpu_fraction": 0.95,
        "task_chunk_size": 10,
        "max_in_flight_factor": 2,
        "checkpoint_every": 25,
        "heartbeat_percent": 0.5,
        "heartbeat_alive_seconds": 30.0,
    },
    "validation": {
        "verify_outputs": True,
        "verify_hashes_on_resume": True,
    },
    "output": {
        "layout": "by_family_recipe",
    },
}


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _reject_unknown_keys(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    for key, value in candidate.items():
        qualified = f"{prefix}.{key}" if prefix else key
        if key not in reference:
            raise ValueError(f"Unknown augmentation configuration key: {qualified}")
        if isinstance(value, dict):
            expected = reference[key]
            if not isinstance(expected, dict):
                raise ValueError(f"Configuration key {qualified} must not be a table.")
            _reject_unknown_keys(expected, value, prefix=qualified)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer; received {value!r}.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}; received {value}.")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric; received {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; received {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false; received {value!r}.")
    return value


def validate_config(config: dict[str, Any]) -> None:
    run = config["run"]
    if not isinstance(run["name"], str) or not run["name"].strip():
        raise ValueError("run.name must be a non-empty string.")
    sources = run["sources"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(value, str) or not value.strip() for value in sources)
    ):
        raise ValueError(
            "run.sources must be a non-empty list of file or folder paths."
        )
    if not isinstance(run["output_root"], str) or not run["output_root"].strip():
        raise ValueError("run.output_root must be a non-empty path.")
    _boolean(run["resume"], "run.resume")

    generation = config["generation"]
    _integer(generation["count_per_recipe"], "generation.count_per_recipe", minimum=1)
    _integer(generation["start_sample"], "generation.start_sample", minimum=0)
    _integer(generation["base_seed"], "generation.base_seed", minimum=0)
    center_min = _finite_float(generation["center_min"], "generation.center_min")
    center_max = _finite_float(generation["center_max"], "generation.center_max")
    beta_min = _finite_float(generation["beta_min"], "generation.beta_min")
    beta_max = _finite_float(generation["beta_max"], "generation.beta_max")
    beta_gap = _finite_float(generation["beta_gap"], "generation.beta_gap")
    beta_opt_gap = _finite_float(
        generation["beta_opt_gap"], "generation.beta_opt_gap"
    )
    if center_min > center_max:
        raise ValueError("generation.center_min must not exceed center_max.")
    if beta_min > beta_max:
        raise ValueError("generation.beta_min must not exceed beta_max.")
    if beta_gap < 0.0:
        raise ValueError("generation.beta_gap must be nonnegative.")
    if beta_gap > beta_max - beta_min:
        raise ValueError(
            "generation.beta_gap must not exceed beta_max - beta_min; "
            "otherwise the requested depth span cannot be honored exactly."
        )
    if not 0.0 <= beta_opt_gap <= beta_gap:
        raise ValueError(
            "generation.beta_opt_gap must lie between zero and beta_gap."
        )
    _boolean(generation["ensure_no_cover"], "generation.ensure_no_cover")
    _integer(
        generation["no_cover_max_attempts"],
        "generation.no_cover_max_attempts",
        minimum=1,
    )
    if (
        _finite_float(
            generation["no_cover_tolerance"], "generation.no_cover_tolerance"
        )
        < 0.0
    ):
        raise ValueError("generation.no_cover_tolerance must be nonnegative.")
    _boolean(generation["trace_metadata"], "generation.trace_metadata")

    execution = config["execution"]
    _integer(execution["max_workers"], "execution.max_workers", minimum=0)
    cpu_fraction = _finite_float(execution["cpu_fraction"], "execution.cpu_fraction")
    if not 0.0 < cpu_fraction <= 1.0:
        raise ValueError("execution.cpu_fraction must lie in (0, 1].")
    _integer(execution["task_chunk_size"], "execution.task_chunk_size", minimum=1)
    _integer(
        execution["max_in_flight_factor"],
        "execution.max_in_flight_factor",
        minimum=1,
    )
    _integer(execution["checkpoint_every"], "execution.checkpoint_every", minimum=1)
    heartbeat_percent = _finite_float(
        execution["heartbeat_percent"], "execution.heartbeat_percent"
    )
    if not 0.0 < heartbeat_percent <= 100.0:
        raise ValueError("execution.heartbeat_percent must lie in (0, 100].")
    if (
        _finite_float(
            execution["heartbeat_alive_seconds"],
            "execution.heartbeat_alive_seconds",
        )
        < 0.0
    ):
        raise ValueError("execution.heartbeat_alive_seconds must be nonnegative.")

    validation = config["validation"]
    _boolean(validation["verify_outputs"], "validation.verify_outputs")
    _boolean(
        validation["verify_hashes_on_resume"],
        "validation.verify_hashes_on_resume",
    )

    layout = config["output"]["layout"]
    if layout not in {"by_family_recipe", "by_recipe", "flat"}:
        raise ValueError(
            "output.layout must be one of: by_family_recipe, by_recipe, flat."
        )


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Load TOML, validate it, and resolve paths from the repository root."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Augmentation configuration not found: {source}")
    with source.open("rb") as handle:
        user_config = tomllib.load(handle)
    _reject_unknown_keys(DEFAULT_CONFIG, user_config)
    config = _merge(DEFAULT_CONFIG, user_config)
    validate_config(config)

    root = Path(project_root).resolve()
    resolved_sources: list[str] = []
    for raw in config["run"]["sources"]:
        value = Path(raw).expanduser()
        if not value.is_absolute():
            value = root / value
        resolved_sources.append(str(value.resolve()))
    output = Path(config["run"]["output_root"]).expanduser()
    if not output.is_absolute():
        output = root / output
    config["run"]["sources"] = resolved_sources
    config["run"]["output_root"] = str(output.resolve())
    config["_meta"] = {
        "config_path": str(source),
        "project_root": str(root),
    }
    return config


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-compatible copy used in run provenance."""

    return json.loads(json.dumps(config, sort_keys=True))
