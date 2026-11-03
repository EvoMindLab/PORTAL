"""Orchestrate the complete PORTAL suite-construction workflow."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import serializable_config
from Recipe_Selection._terminology import normalize_performance_keys
from .discovery import (
    RecipeRecord,
    discover_recipes,
    recipe_fingerprint,
    write_recipe_manifest,
)


BASE_STAGES = [
    "validate",
    "ela",
    "performance",
    "prepare",
    "deduplicate",
    "select",
    "materialize",
]
OPTIONAL_TRAIN_TEST_STAGES = ["train_test", "materialize_train_test"]


@dataclass(frozen=True)
class StagePlan:
    name: str
    commands: tuple[tuple[str, ...], ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    environment: tuple[tuple[str, str], ...] = ()
    sources: tuple[Path, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def _tau_token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _package_versions() -> dict[str, str | None]:
    """Return versions that can materially affect numerical pipeline outputs."""

    versions: dict[str, str | None] = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pulp",
        "pflacco",
        "scikit-learn",
        "threadpoolctl",
        "torch",
        "cma",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # pflacco is vendored in this repository rather than installed as
            # a versioned distribution. Make that provenance explicit.
            if package == "pflacco":
                try:
                    importlib.import_module("pflacco")
                except ImportError:
                    versions[package] = None
                else:
                    versions[package] = "vendored-local"
            else:
                versions[package] = None
    return versions


class SuitePipelineRunner:
    """Execute configured stages with validation, logging, and resume support."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: str | Path,
        dry_run: bool = False,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force: bool = False,
        force_stages: Iterable[str] = (),
    ) -> None:
        self.config = normalize_performance_keys(config)
        self.project_root = Path(project_root).resolve()
        self.recipe_root = Path(config["run"]["recipes_root"]).resolve()
        self.output_root = Path(config["run"]["output_root"]).resolve()
        self.dry_run = bool(dry_run)
        self.from_stage = "performance" if from_stage == "isa" else from_stage
        self.to_stage = "performance" if to_stage == "isa" else to_stage
        self.force = bool(force)
        self.force_stages = {"performance" if name == "isa" else name for name in force_stages}
        self.python = Path(sys.executable).resolve()
        self.script_dir = self.project_root / "Recipe_Selection"
        self.paths = self._build_paths()
        self.records: list[RecipeRecord] = []
        self.input_fingerprint = ""
        self.state: dict[str, Any] = {}

    @property
    def stage_names(self) -> list[str]:
        names = list(BASE_STAGES)
        if bool(self.config["train_test"].get("enabled", False)):
            names.extend(OPTIONAL_TRAIN_TEST_STAGES)
        return names

    def _build_paths(self) -> dict[str, Path]:
        root = self.output_root
        selection_stem = f"benchmark{int(self.config['selection']['size'])}"
        return {
            "metadata": root / "metadata",
            "logs": root / "logs",
            "pipeline_log": root / "logs" / "pipeline.log",
            "raw": root / "raw_features",
            "features": root / "cleaned_features",
            "dedup": root / "deduplication",
            "selection": root / "selection",
            "suite": root / "selected_recipes",
            "train_test": root / "train_test",
            "train_suite": root / "train_test" / "train_recipes",
            "test_suite": root / "train_test" / "test_recipes",
            "state": root / "metadata" / "pipeline_state.json",
            "manifest": root / "metadata" / "run_manifest.json",
            "recipes": root / "metadata" / "recipe_manifest.csv",
            "config": root / "metadata" / "config.resolved.json",
            "ela_raw": root / "raw_features" / "ela_raw.csv",
            "ela_metadata": root / "raw_features" / "ela_metadata.md",
            "ela_cache": root / "raw_features" / "cache",
            "performance_raw": root / "raw_features" / "performance_raw.csv",
            "performance_partial": root / "raw_features" / "performance_runs.partial.csv",
            "performance_metadata": root / "raw_features" / "performance_metadata.md",
            "performance_cleaned": root / "cleaned_features" / "performance_cleaned.csv",
            "ela_cleaned": root / "cleaned_features" / "ela_cleaned.csv",
            "selection_csv": root / "selection" / f"{selection_stem}_selected.csv",
            "selection_report": (
                root / "selection" / f"{selection_stem}_selection_report.md"
            ),
            "train_csv": root / "train_test" / "train_selected.csv",
            "test_csv": root / "train_test" / "test_selected.csv",
        }

    def preflight(self, *, require_cbc: bool) -> None:
        if self.output_root == self.recipe_root or self.output_root.is_relative_to(
            self.recipe_root
        ):
            raise ValueError(
                "run.output_root must be outside run.recipes_root; otherwise generated JSONs "
                "could be rediscovered as input recipes."
            )
        for script in (
            "run_ela.py",
            "run_curation_solvers.py",
            "portal_feature_pipeline.py",
            "run_near_duplicate_tevc.py",
            "run_benchmark_selection.py",
            "sync_selected_suite.py",
        ):
            if not (self.script_dir / script).exists():
                raise FileNotFoundError(
                    f"Required pipeline stage is missing: {self.script_dir / script}"
                )

        self.records = discover_recipes(self.recipe_root, require_single_dimension=True)
        self.input_fingerprint = recipe_fingerprint(self.records)
        # Use the same canonical loader consumed by script-based evaluation and
        # PORTAL Studio. Metadata discovery alone cannot validate component,
        # transformation, block-index, and neutralization fields.
        from portal_core.load_instance import load_instance_payload

        for record in self.records:
            try:
                payload = json.loads(record.path.read_text(encoding="utf-8-sig"))
                runtime = load_instance_payload(payload)
            except Exception as exc:
                raise ValueError(
                    f"Unified recipe failed canonical runtime loading: {record.path}: {exc}"
                ) from exc
            if int(runtime["d"]) != record.dimension:
                raise ValueError(
                    f"Runtime dimension mismatch for {record.path}: "
                    f"metadata={record.dimension}, runtime={runtime['d']}"
                )
        configured_families = [
            str(value) for value in self.config["selection"]["families"]
        ]
        discovered_families = {record.family for record in self.records}
        unconfigured = sorted(discovered_families - set(configured_families))
        if unconfigured:
            raise ValueError(
                f"Recipe JSONs contain families absent from selection.families: {unconfigured}"
            )
        lower = self.config["selection"]["family_lower_percent"]
        for index, family in enumerate(configured_families):
            if family not in discovered_families and float(lower[index]) > 0.0:
                raise ValueError(
                    f"Family {family!r} has a positive lower bound but no input recipes."
                )

        dimension = self.records[0].dimension
        maximum_fe = int(self.config["performance"]["max_fe_per_dimension"]) * dimension
        too_large = [
            int(value)
            for value in self.config["performance"]["checkpoints"]
            if int(value) > maximum_fe
        ]
        if too_large:
            raise ValueError(
                f"Performance checkpoints {too_large} exceed the configured budget {maximum_fe} "
                f"for d={dimension}."
            )

        from Recipe_Selection.run_curation_solvers import CONFIG as performance_stage_config

        supported_algorithms = {
            str(specification["file"]) for specification in performance_stage_config["SOLVERS"]
        }
        configured_algorithms = set(self.config["performance"]["algorithms"])
        unsupported_algorithms = sorted(configured_algorithms - supported_algorithms)
        if unsupported_algorithms:
            raise ValueError(
                "Configured Performance algorithms are not in the curation portfolio: "
                f"{unsupported_algorithms}. Supported values: "
                f"{sorted(supported_algorithms)}"
            )
        available_algorithms = {
            path.stem for path in (self.project_root / "algorithms").glob("*.py")
        }
        missing_algorithms = sorted(configured_algorithms - available_algorithms)
        if missing_algorithms:
            raise FileNotFoundError(
                f"Configured Performance algorithm modules are missing: {missing_algorithms}"
            )

        if require_cbc:
            try:
                import pulp
            except ImportError as exc:
                raise RuntimeError(
                    "The de-duplication stage requires PuLP with its CBC backend."
                ) from exc
            if not pulp.PULP_CBC_CMD(msg=0).available():
                raise RuntimeError(
                    "PuLP is installed, but its CBC backend is unavailable."
                )

        if int(self.config["selection"]["size"]) > len(self.records):
            raise ValueError("selection.size exceeds the number of input recipes.")

    def _stage_environment(self) -> dict[str, str]:
        return {
            "PORTAL_FAMILY_ORDER": _csv(self.config["selection"]["families"]),
        }

    def _build_stage_plans(self) -> dict[str, StagePlan]:
        python = str(self.python)
        ela = self.config["ela"]
        performance = self.config["performance"]
        features = self.config["features"]
        dedup = self.config["deduplication"]
        selection = self.config["selection"]
        env_base = self._stage_environment()
        pipeline_sources = tuple(
            sorted(
                {
                    Path(__file__).resolve(),
                    (Path(__file__).resolve().parent / "config.py").resolve(),
                    (Path(__file__).resolve().parent / "discovery.py").resolve(),
                    (self.script_dir / "_execution.py").resolve(),
                    (self.script_dir / "_terminology.py").resolve(),
                },
                key=lambda path: str(path).casefold(),
            )
        )
        portal_core_sources = tuple(
            sorted(
                (self.project_root / "portal_core").rglob("*.py"),
                key=lambda path: str(path).casefold(),
            )
        )
        pflacco_sources = tuple(
            sorted(
                (self.project_root / "pflacco").rglob("*.py"),
                key=lambda path: str(path).casefold(),
            )
        )
        algorithm_sources = tuple(
            (self.project_root / "algorithms" / f"{name}.py").resolve()
            for name in performance["algorithms"]
        ) + (
            (self.project_root / "algorithms" / "base_algorithm.py").resolve(),
            (self.project_root / "portal_problem.py").resolve(),
        )
        primary_pool = self.paths["dedup"] / (
            f"mis_pool_tau_{_tau_token(float(dedup['primary_tau']))}.csv"
        )

        ela_command = [
            python,
            str(self.script_dir / "run_ela.py"),
            "--instances-root",
            str(self.recipe_root),
            "--output-csv",
            str(self.paths["ela_raw"]),
            "--metadata-md",
            str(self.paths["ela_metadata"]),
            "--sample-factor",
            str(ela["sample_factor"]),
            "--ela-seed",
            str(ela["seed"]),
            "--heartbeat-percent",
            str(ela["heartbeat_percent"]),
            "--heartbeat-alive-seconds",
            str(ela["heartbeat_alive_seconds"]),
            "--evaluation-device",
            str(ela["evaluation_device"]),
        ]
        if int(ela["max_workers"]) > 0:
            ela_command.extend(["--max-workers", str(ela["max_workers"])])

        performance_command = [
            python,
            str(self.script_dir / "run_curation_solvers.py"),
            "--instances-root",
            str(self.recipe_root),
            "--output-csv",
            str(self.paths["performance_raw"]),
            "--checkpoint-csv",
            str(self.paths["performance_partial"]),
            "--metadata-md",
            str(self.paths["performance_metadata"]),
            "--algorithms",
            _csv(performance["algorithms"]),
            "--n-runs",
            str(performance["runs"]),
            "--max-fe-per-dim",
            str(performance["max_fe_per_dimension"]),
            "--checkpoints-fe",
            _csv(performance["checkpoints"]),
            "--base-run-seed",
            str(performance["base_seed"]),
            "--checkpoint-every",
            str(performance["checkpoint_every"]),
            "--heartbeat-percent",
            str(performance["heartbeat_percent"]),
            "--heartbeat-alive-seconds",
            str(performance["heartbeat_alive_seconds"]),
            "--evaluation-device",
            str(performance["evaluation_device"]),
            "--resume" if bool(performance["resume"]) else "--no-resume",
        ]
        if int(performance["max_parallel_runs"]) > 0:
            performance_command.extend(["--max-parallel-runs", str(performance["max_parallel_runs"])])

        prepare_command = [
            python,
            str(self.script_dir / "portal_feature_pipeline.py"),
            "--performance_file",
            str(self.paths["performance_raw"]),
            "--ela_file",
            str(self.paths["ela_raw"]),
            "--out_dir",
            str(self.paths["features"]),
            "--log_eps",
            str(features["log_epsilon"]),
            "--iqr_floor",
            str(features["iqr_floor"]),
            "--corr_threshold",
            str(features["correlation_threshold"]),
            "--mad_floor",
            str(features["mad_floor"]),
            "--winsor_low",
            str(features["winsor_lower"]),
            "--winsor_high",
            str(features["winsor_upper"]),
            "--algorithms",
            _csv(performance["algorithms"]),
            "--checkpoints",
            _csv(performance["checkpoints"]),
        ]

        dedup_env = {
            **env_base,
            "PORTAL_PIPELINE_DIR": str(self.paths["features"]),
            "PORTAL_NEAR_DUP_OUT_DIR": str(self.paths["dedup"]),
            "PORTAL_DEDUP_THRESHOLDS": _csv(dedup["sensitivity_thresholds"]),
            "PORTAL_PRIMARY_TAU": str(dedup["primary_tau"]),
            "PORTAL_MIS_BACKEND": str(dedup["mis_backend"]),
        }
        dedup_command = [python, str(self.script_dir / "run_near_duplicate_tevc.py")]

        selection_command = [
            python,
            str(self.script_dir / "run_benchmark_selection.py"),
            "--pool-file",
            str(primary_pool),
            "--performance-cleaned",
            str(self.paths["performance_cleaned"]),
            "--ela-cleaned",
            str(self.paths["ela_cleaned"]),
            "--k",
            str(selection["size"]),
            "--w-ela",
            str(selection["ela_weight"]),
            "--w-performance",
            str(selection["performance_weight"]),
            "--trim-fraction",
            str(selection["trim_fraction"]),
            "--family-lower",
            _csv(selection["family_lower_percent"]),
            "--family-upper",
            _csv(selection["family_upper_percent"]),
            "--spacing-lambda",
            str(selection["spacing_lambda"]),
            "--spacing-mu",
            str(selection["spacing_mu"]),
            "--max-swap-passes",
            str(selection["max_swap_passes"]),
            "--out-dir",
            str(self.paths["selection"]),
        ]

        materialize_command = [
            python,
            str(self.script_dir / "sync_selected_suite.py"),
            "--selection-csv",
            str(self.paths["selection_csv"]),
            "--suite-root",
            str(self.recipe_root),
            "--target-dir",
            str(self.paths["suite"]),
        ]

        plans = {
            "validate": StagePlan(
                "validate",
                (),
                (),
                (self.paths["recipes"], self.paths["config"]),
                sources=pipeline_sources,
            ),
            "ela": StagePlan(
                "ela",
                (tuple(ela_command),),
                (),
                (self.paths["ela_raw"], self.paths["ela_metadata"]),
                sources=(
                    (self.script_dir / "run_ela.py").resolve(),
                    *pipeline_sources,
                    *portal_core_sources,
                    *pflacco_sources,
                ),
            ),
            "performance": StagePlan(
                "performance",
                (tuple(performance_command),),
                (),
                (
                    self.paths["performance_raw"],
                    self.paths["performance_partial"],
                    self.paths["performance_metadata"],
                ),
                sources=(
                    (self.script_dir / "run_curation_solvers.py").resolve(),
                    *pipeline_sources,
                    *portal_core_sources,
                    *algorithm_sources,
                ),
            ),
            "prepare": StagePlan(
                "prepare",
                (tuple(prepare_command),),
                (self.paths["ela_raw"], self.paths["performance_raw"]),
                (
                    self.paths["ela_cleaned"],
                    self.paths["performance_cleaned"],
                    self.paths["features"] / "ela_dropped_features.csv",
                    self.paths["features"] / "ela_retained_features.csv",
                    self.paths["features"] / "ela_retained_unscaled.csv",
                    self.paths["features"] / "performance_logmedian_unscaled.csv",
                    self.paths["features"] / "pipeline_metadata.json",
                    self.paths["features"] / "pipeline_report.txt",
                ),
                sources=((self.script_dir / "portal_feature_pipeline.py").resolve(),),
            ),
            "deduplicate": StagePlan(
                "deduplicate",
                (tuple(dedup_command),),
                (self.paths["ela_cleaned"], self.paths["performance_cleaned"]),
                (
                    primary_pool,
                    self.paths["dedup"]
                    / f"confirmed_pairs_tau_{_tau_token(float(dedup['primary_tau']))}.csv",
                    self.paths["dedup"] / "mis_sweep_tau.csv",
                    self.paths["dedup"] / "near_dup_tevc_report.md",
                    self.paths["dedup"] / "removal_reason_tevc.csv",
                    self.paths["dedup"]
                    / f"rule_comparison_tau_{_tau_token(float(dedup['primary_tau']))}.csv",
                    self.paths["dedup"] / "sensitivity_table_tau.csv",
                    self.paths["dedup"] / "tevc_crowding.pdf",
                    self.paths["dedup"] / "tevc_crowding.png",
                    self.paths["dedup"] / "tevc_distance_scatter.pdf",
                    self.paths["dedup"] / "tevc_distance_scatter.png",
                    self.paths["dedup"] / "tevc_sensitivity.pdf",
                    self.paths["dedup"] / "tevc_sensitivity.png",
                ),
                tuple(sorted(dedup_env.items())),
                ((self.script_dir / "run_near_duplicate_tevc.py").resolve(),),
            ),
            "select": StagePlan(
                "select",
                (tuple(selection_command),),
                (primary_pool, self.paths["ela_cleaned"], self.paths["performance_cleaned"]),
                (
                    self.paths["selection_csv"],
                    self.paths["selection_report"],
                ),
                tuple(sorted(env_base.items())),
                ((self.script_dir / "run_benchmark_selection.py").resolve(),),
            ),
            "materialize": StagePlan(
                "materialize",
                (tuple(materialize_command),),
                (self.paths["selection_csv"],),
                (self.paths["suite"],),
                sources=(
                    (self.script_dir / "sync_selected_suite.py").resolve(),
                    *pipeline_sources,
                ),
            ),
        }

        if bool(self.config["train_test"].get("enabled", False)):
            train_test = self.config["train_test"]
            train_test_command = [
                python,
                str(self.script_dir / "run_train_test_selection.py"),
                "--pool-file",
                str(primary_pool),
                "--performance-cleaned",
                str(self.paths["performance_cleaned"]),
                "--ela-cleaned",
                str(self.paths["ela_cleaned"]),
                "--train-k",
                str(train_test["train_size"]),
                "--test-k",
                str(train_test["test_size"]),
                "--train-family-lower",
                _csv(train_test["train_family_lower_percent"]),
                "--train-family-upper",
                _csv(train_test["train_family_upper_percent"]),
                "--test-family-lower",
                _csv(train_test["test_family_lower_percent"]),
                "--test-family-upper",
                _csv(train_test["test_family_upper_percent"]),
                "--split-seed",
                str(train_test["split_seed"]),
                "--w-ela",
                str(selection["ela_weight"]),
                "--w-performance",
                str(selection["performance_weight"]),
                "--trim-fraction",
                str(selection["trim_fraction"]),
                "--spacing-lambda",
                str(selection["spacing_lambda"]),
                "--spacing-mu",
                str(selection["spacing_mu"]),
                "--max-swap-passes",
                str(selection["max_swap_passes"]),
                "--out-dir",
                str(self.paths["train_test"]),
            ]
            train_sync = [
                python,
                str(self.script_dir / "sync_selected_suite.py"),
                "--selection-csv",
                str(self.paths["train_csv"]),
                "--suite-root",
                str(self.recipe_root),
                "--target-dir",
                str(self.paths["train_suite"]),
            ]
            test_sync = [
                python,
                str(self.script_dir / "sync_selected_suite.py"),
                "--selection-csv",
                str(self.paths["test_csv"]),
                "--suite-root",
                str(self.recipe_root),
                "--target-dir",
                str(self.paths["test_suite"]),
            ]
            plans["train_test"] = StagePlan(
                "train_test",
                (tuple(train_test_command),),
                (primary_pool, self.paths["ela_cleaned"], self.paths["performance_cleaned"]),
                (
                    self.paths["train_test"] / "train_pool_half.csv",
                    self.paths["train_test"] / "test_pool_half.csv",
                    self.paths["train_csv"],
                    self.paths["test_csv"],
                    self.paths["train_test"] / "train_test_selected.csv",
                    self.paths["train_test"] / "train_test_selection_report.md",
                ),
                tuple(sorted(env_base.items())),
                (
                    (self.script_dir / "run_train_test_selection.py").resolve(),
                    (self.script_dir / "run_benchmark_selection.py").resolve(),
                ),
            )
            plans["materialize_train_test"] = StagePlan(
                "materialize_train_test",
                (tuple(train_sync), tuple(test_sync)),
                (self.paths["train_csv"], self.paths["test_csv"]),
                (self.paths["train_suite"], self.paths["test_suite"]),
                sources=(
                    (self.script_dir / "sync_selected_suite.py").resolve(),
                    *pipeline_sources,
                ),
            )
        return plans

    def _selected_stage_names(self) -> list[str]:
        names = self.stage_names
        for label, value in (("from", self.from_stage), ("to", self.to_stage)):
            if value is not None and value not in names:
                raise ValueError(f"Unknown --{label}-stage {value!r}; choices: {names}")
        start = names.index(self.from_stage) if self.from_stage else 0
        stop = names.index(self.to_stage) + 1 if self.to_stage else len(names)
        if start >= stop:
            raise ValueError("--from-stage must not occur after --to-stage.")
        return names[start:stop]

    def _load_state(self) -> None:
        path = self.paths["state"]
        if path.exists():
            try:
                self.state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.state = {}
        if not isinstance(self.state, dict):
            self.state = {}
        self.state.setdefault("stages", {})

    def _stage_signature(self, plan: StagePlan) -> str:
        input_hashes: dict[str, str] = {}
        for path in plan.inputs:
            if not path.exists() or not path.is_file():
                input_hashes[str(path)] = "MISSING"
            else:
                input_hashes[str(path)] = _hash_file(path)
        payload = {
            "stage": plan.name,
            "commands": plan.commands,
            "environment": plan.environment,
            "inputs": input_hashes,
            "sources": {
                str(path): (
                    _hash_file(path) if path.exists() and path.is_file() else "MISSING"
                )
                for path in plan.sources
            },
            "python": str(self.python),
            "python_version": sys.version,
            "package_versions": _package_versions(),
            "recipe_fingerprint": (
                self.input_fingerprint
                if plan.name
                in {"validate", "ela", "performance", "materialize", "materialize_train_test"}
                else None
            ),
        }
        return _hash_payload(payload)

    @staticmethod
    def _path_fingerprint(path: Path) -> dict[str, Any]:
        """Fingerprint one declared output, including directory contents."""

        if not path.exists():
            return {"kind": "missing"}
        if path.is_file():
            return {
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": _hash_file(path),
            }

        files = sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: str(candidate.relative_to(path)).casefold(),
        )
        digest = hashlib.sha256()
        entries: list[dict[str, Any]] = []
        for candidate in files:
            relative = candidate.relative_to(path).as_posix()
            file_hash = _hash_file(candidate)
            size = candidate.stat().st_size
            entries.append({"path": relative, "size": size, "sha256": file_hash})
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\0")
        return {
            "kind": "directory",
            "file_count": len(entries),
            "sha256": digest.hexdigest(),
            "files": entries,
        }

    def _output_fingerprints(self, plan: StagePlan) -> dict[str, dict[str, Any]]:
        return {str(path): self._path_fingerprint(path) for path in plan.outputs}

    @staticmethod
    def _outputs_exist(plan: StagePlan) -> bool:
        for path in plan.outputs:
            if not path.exists():
                return False
            if path.is_file() and path.stat().st_size == 0:
                return False
            if path.is_dir():
                json_files = [
                    candidate
                    for candidate in path.rglob("*.json")
                    if candidate.is_file()
                ]
                if not json_files or any(
                    candidate.stat().st_size == 0 for candidate in json_files
                ):
                    return False
        return True

    def _can_resume(self, plan: StagePlan, signature: str) -> bool:
        if not bool(self.config["run"].get("resume", True)):
            return False
        if self.force or plan.name in self.force_stages:
            return False
        record = self.state.get("stages", {}).get(plan.name, {})
        return (
            record.get("status") == "completed"
            and record.get("signature") == signature
            and self._outputs_exist(plan)
            and record.get("output_fingerprints") == self._output_fingerprints(plan)
        )

    def _validate_inputs(self, plan: StagePlan) -> None:
        missing = [str(path) for path in plan.inputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Stage {plan.name!r} is missing required inputs: {missing}"
            )

    def _prepare_performance_partial_checkpoint(self, plan: StagePlan, signature: str) -> None:
        """Keep a performance partial checkpoint only when its provenance is compatible.

        The Performance child process supports run-level recovery after interruption.
        Stage-level invalidation must not, however, allow rows produced from a
        different recipe catalog, algorithm implementation, environment, or
        numerical configuration to leak into a new run.
        """

        if plan.name != "performance" or not self.paths["performance_partial"].exists():
            return

        record = self.state.get("stages", {}).get("performance", {})
        compatible = (
            bool(self.config["run"]["resume"])
            and bool(self.config["performance"]["resume"])
            and not self.force
            and "performance" not in self.force_stages
            and record.get("signature") == signature
        )

        # For a previously completed stage, also require the checkpoint itself
        # to match its recorded content hash. Interrupted/failed stages do not
        # yet have output fingerprints, so their same-signature checkpoint is
        # intentionally retained for run-level recovery.
        if compatible and record.get("status") == "completed":
            expected = record.get("output_fingerprints", {}).get(
                str(self.paths["performance_partial"])
            )
            compatible = expected is not None and expected == self._path_fingerprint(
                self.paths["performance_partial"]
            )

        if compatible:
            message = "Performance run-level checkpoint is compatible and will be resumed."
            print(f"[performance] {message}")
            self._log_event("stage=performance | checkpoint=retained-compatible")
            return

        self.paths["performance_partial"].unlink()
        message = "removed incompatible Performance run-level checkpoint"
        print(f"[performance] {message}: {self.paths['performance_partial']}")
        self._log_event("stage=performance | checkpoint=removed-incompatible")

    def _prepare_ela_cache(self, plan: StagePlan, signature: str) -> None:
        """Discard an ELA cache when the producing stage signature changes."""

        cache = self.paths["ela_cache"]
        if plan.name != "ela" or not cache.exists():
            return
        record = self.state.get("stages", {}).get("ela", {})
        compatible = (
            bool(self.config["run"]["resume"])
            and not self.force
            and "ela" not in self.force_stages
            and record.get("signature") == signature
        )
        if compatible:
            return

        resolved_cache = cache.resolve()
        if resolved_cache == self.output_root or not resolved_cache.is_relative_to(
            self.output_root
        ):
            raise RuntimeError(
                f"Refusing to remove ELA cache outside the run output: {resolved_cache}"
            )
        shutil.rmtree(resolved_cache)
        print(f"[ela] removed incompatible sampling cache: {resolved_cache}")
        self._log_event("stage=ela | cache=removed-incompatible")

    def _run_command(
        self,
        stage: str,
        command: tuple[str, ...],
        environment: dict[str, str],
        command_index: int,
    ) -> None:
        log_path = self.paths["logs"] / f"{stage}.log"
        env = os.environ.copy()
        env.update(environment)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        printable = subprocess.list2cmdline(list(command))
        print(f"[{stage}] {printable}", flush=True)
        command_started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write(
                f"{_utc_now()} | stage={stage} | command={command_index + 1} | start\n"
            )
            log.write(f"$ {printable}\n")
            if environment:
                log.write(
                    f"environment overrides: {json.dumps(environment, sort_keys=True)}\n"
                )
            log.flush()
            process = subprocess.Popen(
                list(command),
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except KeyboardInterrupt:
                log.write(
                    f"{_utc_now()} | stage={stage} | command={command_index + 1} | "
                    "interrupted by user; terminating child process\n"
                )
                log.flush()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            log.write(
                f"{_utc_now()} | stage={stage} | command={command_index + 1} | "
                f"exit_code={return_code} | elapsed_seconds={time.time() - command_started:.3f}\n"
            )
            log.flush()
        if return_code != 0:
            raise RuntimeError(
                f"Stage {stage!r} failed with exit code {return_code}; see {log_path}."
            )

    def _write_validation_outputs(self) -> None:
        write_recipe_manifest(self.records, self.paths["recipes"])
        _write_json_atomic(self.paths["config"], serializable_config(self.config))

    def _record_stage(self, plan: StagePlan, signature: str, started: float) -> None:
        self.state.setdefault("stages", {})[plan.name] = {
            "status": "completed",
            "signature": signature,
            "completed_at": _utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "commands": [list(command) for command in plan.commands],
            "outputs": [str(path) for path in plan.outputs],
            "output_fingerprints": self._output_fingerprints(plan),
        }
        self.state["run_name"] = self.config["run"]["name"]
        self.state["recipe_fingerprint"] = self.input_fingerprint
        _write_json_atomic(self.paths["state"], self.state)

    def _log_event(self, message: str) -> None:
        """Append a timestamped outer-run event to the master pipeline log."""

        if self.dry_run:
            return
        self.paths["logs"].mkdir(parents=True, exist_ok=True)
        with self.paths["pipeline_log"].open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} | {message}\n")

    def _mark_stage_status(
        self,
        plan: StagePlan,
        signature: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        record = {
            "status": status,
            "signature": signature,
            "updated_at": _utc_now(),
            "commands": [list(command) for command in plan.commands],
            "outputs": [str(path) for path in plan.outputs],
        }
        if error is not None:
            record["error"] = error
        self.state.setdefault("stages", {})[plan.name] = record
        self.state["run_name"] = self.config["run"]["name"]
        self.state["recipe_fingerprint"] = self.input_fingerprint
        _write_json_atomic(self.paths["state"], self.state)

    def _git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    def _write_run_manifest(self) -> None:
        family_counts: dict[str, int] = {}
        for record in self.records:
            family_counts[record.family] = family_counts.get(record.family, 0) + 1
        manifest = {
            "generated_at": _utc_now(),
            "run_name": self.config["run"]["name"],
            "project_root": str(self.project_root),
            "python_executable": str(self.python),
            "python_version": sys.version,
            "git_commit": self._git_commit(),
            "recipe_root": str(self.recipe_root),
            "recipe_count": len(self.records),
            "family_counts": family_counts,
            "dimension": self.records[0].dimension,
            "recipe_fingerprint": self.input_fingerprint,
            "package_versions": _package_versions(),
            "config": serializable_config(self.config),
            "stage_state": self.state.get("stages", {}),
        }
        _write_json_atomic(self.paths["manifest"], manifest)

    def run(self) -> None:
        selected = self._selected_stage_names()
        self.preflight(
            require_cbc=("deduplicate" in selected or "train_test" in selected)
        )
        plans = self._build_stage_plans()
        unknown_forced = self.force_stages - set(self.stage_names)
        if unknown_forced:
            raise ValueError(f"Unknown --force-stage values: {sorted(unknown_forced)}")

        print("=" * 72)
        print("PORTAL end-to-end suite-construction pipeline")
        print("=" * 72)
        print(f"Python:       {self.python}")
        print(f"Recipes:      {self.recipe_root}")
        print(f"Recipe count: {len(self.records)}")
        family_counts: dict[str, int] = {}
        for record in self.records:
            family_counts[record.family] = family_counts.get(record.family, 0) + 1
        print(f"Families:     {family_counts}")
        print(f"Dimension:    {self.records[0].dimension}")
        print(f"Output:       {self.output_root}")
        print(f"Stages:       {selected}")
        print(
            "ELA:          "
            f"{int(self.config['ela']['sample_factor'])} x d samples; "
            f"workers={self.config['ela']['max_workers'] or 'automatic (90% CPU)'}"
        )
        print(
            "Performance:          "
            f"{len(self.config['performance']['algorithms'])} algorithms x "
            f"{int(self.config['performance']['runs'])} runs x {len(self.records)} recipes; "
            f"parallel={self.config['performance']['max_parallel_runs'] or 'automatic (90% CPU)'}"
        )
        print(
            "Selection:    "
            f"K={int(self.config['selection']['size'])}; "
            f"family lower%={self.config['selection']['family_lower_percent']}; "
            f"upper%={self.config['selection']['family_upper_percent']}"
        )
        if bool(self.config["train_test"].get("enabled", False)):
            print(
                "Train/test:   "
                f"train={int(self.config['train_test']['train_size'])}, "
                f"test={int(self.config['train_test']['test_size'])}"
            )
        if self.dry_run:
            print("Mode:         dry run (no files or subprocesses will be created)")

        if not self.dry_run:
            for path in (self.paths["metadata"], self.paths["logs"]):
                path.mkdir(parents=True, exist_ok=True)
            self._load_state()
            # Refresh provenance even when a partial run starts after validation.
            self._write_validation_outputs()
            self._log_event(
                f"run start | name={self.config['run']['name']} | "
                f"recipes={len(self.records)} | stages={','.join(selected)}"
            )

        for name in selected:
            plan = plans[name]
            signature = self._stage_signature(plan)
            if self.dry_run:
                if plan.commands:
                    for command in plan.commands:
                        print(f"[{name}] {subprocess.list2cmdline(list(command))}")
                else:
                    print(f"[{name}] validate {len(self.records)} unified recipe JSONs")
                continue
            self._validate_inputs(plan)
            if self._can_resume(plan, signature):
                print(f"[{name}] resume: compatible outputs already exist; skipped.")
                self._log_event(f"stage={name} | status=skipped-compatible")
                continue

            self._prepare_ela_cache(plan, signature)
            self._prepare_performance_partial_checkpoint(plan, signature)

            for output in plan.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self._mark_stage_status(plan, signature, "running")
            self._log_event(f"stage={name} | status=running")
            try:
                if name == "validate":
                    # Provenance was written before stage execution so partial runs
                    # receive the same metadata as complete runs.
                    pass
                else:
                    environment = dict(plan.environment)
                    for index, command in enumerate(plan.commands):
                        self._run_command(name, command, environment, index)
                if not self._outputs_exist(plan):
                    missing = [str(path) for path in plan.outputs if not path.exists()]
                    raise RuntimeError(
                        f"Stage {name!r} completed without all expected outputs: {missing}"
                    )
            except KeyboardInterrupt:
                self._mark_stage_status(
                    plan,
                    signature,
                    "interrupted",
                    error="Interrupted by user",
                )
                self._log_event(f"stage={name} | status=interrupted-by-user")
                self._write_run_manifest()
                raise
            except Exception as exc:
                self._mark_stage_status(plan, signature, "failed", error=str(exc))
                self._log_event(f"stage={name} | status=failed | error={exc}")
                self._write_run_manifest()
                raise
            self._record_stage(plan, signature, started)
            self._write_run_manifest()
            print(f"[{name}] completed in {time.time() - started:.1f} s")
            self._log_event(
                f"stage={name} | status=completed | elapsed_seconds={time.time() - started:.3f}"
            )

        if not self.dry_run:
            self._write_run_manifest()
            self._log_event("run status=completed")
            print("=" * 72)
            print(f"REQUESTED STAGES COMPLETE: {self.output_root}")
            if self.paths["suite"].exists():
                print(f"Selected suite:           {self.paths['suite']}")
            if self.paths["train_suite"].exists():
                print(f"Selected training suite:  {self.paths['train_suite']}")
            if self.paths["test_suite"].exists():
                print(f"Selected test suite:      {self.paths['test_suite']}")
            print(f"Run manifest:      {self.paths['manifest']}")
