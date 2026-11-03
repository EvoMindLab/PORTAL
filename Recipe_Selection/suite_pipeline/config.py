"""Configuration loading and validation for the PORTAL suite runner."""

from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path
from typing import Any
from Recipe_Selection._terminology import normalize_performance_keys


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "portal_suite",
        "recipes_root": "Suite",
        "output_root": "suite_runs/portal_suite",
        "resume": True,
    },
    "ela": {
        "sample_factor": 1000,
        "seed": 42,
        "max_workers": 0,
        "heartbeat_percent": 0.1,
        "heartbeat_alive_seconds": 30.0,
        "evaluation_device": "cpu",
    },
    "performance": {
        "algorithms": [
            "CMA_ES",
            "DE",
            "LSHADE",
            "PSO",
            "CLPSO",
            "PatternSearch",
            "Powell",
        ],
        "runs": 11,
        "max_fe_per_dimension": 10000,
        "checkpoints": [30000, 50000, 100000, 200000, 300000],
        "base_seed": 10000,
        "max_parallel_runs": 0,
        "checkpoint_every": 25,
        "heartbeat_percent": 0.1,
        "heartbeat_alive_seconds": 30.0,
        "evaluation_device": "cpu",
        "resume": True,
    },
    "features": {
        "log_epsilon": 1e-10,
        "iqr_floor": 1e-6,
        "correlation_threshold": 0.95,
        "mad_floor": 1e-8,
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
    },
    "deduplication": {
        "rule": "and",
        "primary_tau": 0.05,
        "sensitivity_thresholds": [0.03, 0.05, 0.08, 0.10],
        "normalization": "p95",
        "mis_backend": "cbc",
    },
    "selection": {
        "size": 50,
        "families": ["Family1", "Family2", "Family3", "Family4"],
        "family_lower_percent": [12.0, 16.0, 20.0, 30.0],
        "family_upper_percent": [16.0, 24.0, 30.0, 40.0],
        "ela_weight": 0.5,
        "performance_weight": 0.5,
        "trim_fraction": 0.10,
        "spacing_lambda": 1.0,
        "spacing_mu": 0.5,
        "max_swap_passes": 10,
    },
    "train_test": {
        "enabled": False,
        "train_size": 30,
        "test_size": 30,
        "split_seed": 42,
        "train_family_lower_percent": [12.0, 16.0, 20.0, 30.0],
        "train_family_upper_percent": [16.0, 24.0, 30.0, 40.0],
        "test_family_lower_percent": [12.0, 16.0, 20.0, 30.0],
        "test_family_upper_percent": [16.0, 24.0, 30.0, 40.0],
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
    """Reject misspelled or unsupported TOML keys before defaults are merged."""

    for key, value in candidate.items():
        qualified = f"{prefix}.{key}" if prefix else key
        if key not in reference:
            raise ValueError(f"Unknown pipeline configuration key: {qualified}")
        if isinstance(value, dict):
            expected = reference[key]
            if not isinstance(expected, dict):
                raise ValueError(f"Configuration key {qualified} must not be a table.")
            _reject_unknown_keys(expected, value, prefix=qualified)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer; received {value!r}.")
    parsed = value
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {relation}; received {parsed}.")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false; received {value!r}.")
    return value


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    return parsed


def _percentages(values: Any, families: list[str], label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != len(families):
        raise ValueError(f"{label} must provide one value for each family {families}.")
    parsed = [_finite_float(value, label) for value in values]
    if any(value < 0.0 or value > 100.0 for value in parsed):
        raise ValueError(f"{label} values must lie in [0, 100]; received {parsed}.")
    return parsed


def validate_config(config: dict[str, Any]) -> None:
    evaluation_devices = {"auto", "numpy", "cpu", "cuda", "mps"}
    run = config["run"]
    if not str(run.get("name", "")).strip():
        raise ValueError("run.name must not be empty.")
    for key in ("recipes_root", "output_root"):
        if not str(run.get(key, "")).strip():
            raise ValueError(f"run.{key} must not be empty.")
    _boolean(run["resume"], "run.resume")

    ela = config["ela"]
    _positive_int(ela["sample_factor"], "ela.sample_factor")
    _positive_int(ela["max_workers"], "ela.max_workers", allow_zero=True)
    _positive_int(ela["seed"], "ela.seed", allow_zero=True)
    ela_heartbeat = _finite_float(ela["heartbeat_percent"], "ela.heartbeat_percent")
    if not 0.0 < ela_heartbeat <= 100.0:
        raise ValueError("ela.heartbeat_percent must lie in (0, 100].")
    if (
        _finite_float(ela["heartbeat_alive_seconds"], "ela.heartbeat_alive_seconds")
        < 0.0
    ):
        raise ValueError("ela.heartbeat_alive_seconds must be nonnegative.")
    if str(ela["evaluation_device"]) not in evaluation_devices:
        raise ValueError(
            f"ela.evaluation_device must be one of {sorted(evaluation_devices)}."
        )

    performance = config["performance"]
    algorithms = performance.get("algorithms")
    if (
        not isinstance(algorithms, list)
        or not algorithms
        or any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in algorithms
        )
        or len(algorithms) != len(set(algorithms))
    ):
        raise ValueError("performance.algorithms must be a non-empty list without duplicates.")
    _positive_int(performance["runs"], "performance.runs")
    _positive_int(performance["max_fe_per_dimension"], "performance.max_fe_per_dimension")
    _positive_int(performance["max_parallel_runs"], "performance.max_parallel_runs", allow_zero=True)
    _positive_int(performance["checkpoint_every"], "performance.checkpoint_every")
    _positive_int(performance["base_seed"], "performance.base_seed", allow_zero=True)
    performance_heartbeat = _finite_float(performance["heartbeat_percent"], "performance.heartbeat_percent")
    if not 0.0 < performance_heartbeat <= 100.0:
        raise ValueError("performance.heartbeat_percent must lie in (0, 100].")
    if (
        _finite_float(performance["heartbeat_alive_seconds"], "performance.heartbeat_alive_seconds")
        < 0.0
    ):
        raise ValueError("performance.heartbeat_alive_seconds must be nonnegative.")
    if str(performance["evaluation_device"]) not in evaluation_devices:
        raise ValueError(
            f"performance.evaluation_device must be one of {sorted(evaluation_devices)}."
        )
    _boolean(performance["resume"], "performance.resume")
    if not isinstance(performance["checkpoints"], list):
        raise ValueError("performance.checkpoints must be a list of FE values.")
    checkpoints = [
        _positive_int(value, f"performance.checkpoints[{index}]")
        for index, value in enumerate(performance["checkpoints"])
    ]
    if not checkpoints or any(value < 1 for value in checkpoints):
        raise ValueError("performance.checkpoints must contain positive FE values.")
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("performance.checkpoints must be unique and strictly increasing.")

    features = config["features"]
    if _finite_float(features["log_epsilon"], "features.log_epsilon") <= 0.0:
        raise ValueError("features.log_epsilon must be positive.")
    if _finite_float(features["iqr_floor"], "features.iqr_floor") < 0.0:
        raise ValueError("features.iqr_floor must be nonnegative.")
    if _finite_float(features["mad_floor"], "features.mad_floor") <= 0.0:
        raise ValueError("features.mad_floor must be positive.")
    winsor_lower = _finite_float(features["winsor_lower"], "features.winsor_lower")
    winsor_upper = _finite_float(features["winsor_upper"], "features.winsor_upper")
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise ValueError(
            "Feature winsorization bounds must satisfy 0 <= lower < upper <= 1."
        )
    correlation_threshold = _finite_float(
        features["correlation_threshold"], "features.correlation_threshold"
    )
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("features.correlation_threshold must lie in (0, 1].")

    dedup = config["deduplication"]
    if str(dedup["rule"]) != "and":
        raise ValueError(
            "The curation runner currently supports the conservative AND rule only."
        )
    if str(dedup["normalization"]) != "p95":
        raise ValueError(
            "The de-duplication stage requires P95 distance normalization."
        )
    if str(dedup["mis_backend"]) != "cbc":
        raise ValueError(
            "The de-duplication stage currently requires the CBC MIS backend."
        )
    if not isinstance(dedup["sensitivity_thresholds"], list):
        raise ValueError("deduplication.sensitivity_thresholds must be a list.")
    thresholds = [
        _finite_float(value, "deduplication.sensitivity_thresholds")
        for value in dedup["sensitivity_thresholds"]
    ]
    primary = _finite_float(dedup["primary_tau"], "deduplication.primary_tau")
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("De-duplication thresholds must lie in (0, 1].")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("deduplication.sensitivity_thresholds must be unique.")
    if not 0.0 < primary <= 1.0:
        raise ValueError("deduplication.primary_tau must lie in (0, 1].")
    if not any(abs(value - primary) <= 1e-12 for value in thresholds):
        raise ValueError(
            "deduplication.primary_tau must occur in sensitivity_thresholds."
        )

    selection = config["selection"]
    _positive_int(selection["size"], "selection.size")
    if not isinstance(selection["families"], list):
        raise ValueError("selection.families must be a list.")
    families = [str(value).strip() for value in selection["families"]]
    if (
        not families
        or any(
            not family or "," in family or str(raw) != str(raw).strip()
            for raw, family in zip(selection["families"], families, strict=True)
        )
        or len(families) != len({family.casefold() for family in families})
    ):
        raise ValueError("selection.families must be non-empty and unique.")
    lower = _percentages(
        selection["family_lower_percent"], families, "family_lower_percent"
    )
    upper = _percentages(
        selection["family_upper_percent"], families, "family_upper_percent"
    )
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError(
            "Each family lower percentage must not exceed its upper percentage."
        )
    if sum(lower) > 100.0 + 1e-9 or sum(upper) < 100.0 - 1e-9:
        raise ValueError(
            "Family percentage bounds cannot contain the requested suite cardinality."
        )
    ela_weight = _finite_float(selection["ela_weight"], "selection.ela_weight")
    performance_weight = _finite_float(selection["performance_weight"], "selection.performance_weight")
    if ela_weight < 0 or performance_weight < 0:
        raise ValueError("Selection distance weights must be nonnegative.")
    if ela_weight + performance_weight <= 0:
        raise ValueError("At least one selection distance weight must be positive.")
    trim_fraction = _finite_float(selection["trim_fraction"], "selection.trim_fraction")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("selection.trim_fraction must lie in [0, 0.5).")
    spacing_lambda = _finite_float(
        selection["spacing_lambda"], "selection.spacing_lambda"
    )
    spacing_mu = _finite_float(selection["spacing_mu"], "selection.spacing_mu")
    if spacing_lambda < 0.0 or spacing_mu < 0.0:
        raise ValueError("Selection spacing weights must be nonnegative.")
    _positive_int(
        selection["max_swap_passes"], "selection.max_swap_passes", allow_zero=True
    )

    train_test = config["train_test"]
    _boolean(train_test["enabled"], "train_test.enabled")
    if train_test["enabled"]:
        _positive_int(train_test["train_size"], "train_test.train_size")
        _positive_int(train_test["test_size"], "train_test.test_size")
        _positive_int(
            train_test["split_seed"], "train_test.split_seed", allow_zero=True
        )
        for side in ("train", "test"):
            lo = _percentages(
                train_test[f"{side}_family_lower_percent"],
                families,
                f"train_test.{side}_family_lower_percent",
            )
            hi = _percentages(
                train_test[f"{side}_family_upper_percent"],
                families,
                f"train_test.{side}_family_upper_percent",
            )
            if any(a > b for a, b in zip(lo, hi)):
                raise ValueError(f"Infeasible {side} family percentage bounds.")
            if sum(lo) > 100.0 + 1e-9 or sum(hi) < 100.0 - 1e-9:
                raise ValueError(
                    f"The {side} family percentage bounds cannot contain "
                    f"the requested {side} cardinality."
                )


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Load TOML configuration and resolve paths relative to the project root."""

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Pipeline configuration not found: {source}")
    with source.open("rb") as handle:
        user_config = normalize_performance_keys(tomllib.load(handle))
    _reject_unknown_keys(DEFAULT_CONFIG, user_config)
    config = _merge(DEFAULT_CONFIG, user_config)
    validate_config(config)

    root = Path(project_root).resolve()
    for key in ("recipes_root", "output_root"):
        value = Path(str(config["run"][key])).expanduser()
        if not value.is_absolute():
            value = root / value
        config["run"][key] = str(value.resolve())
    config["_meta"] = {
        "config_path": str(source),
        "project_root": str(root),
    }
    return config


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a stable copy suitable for manifests and hashing."""

    return json.loads(json.dumps(config, sort_keys=True))
