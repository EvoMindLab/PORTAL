"""Resumable, parallel Type 3 dataset augmentation for PORTAL."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from portal_core.fitness import fitness
from portal_core.instance_format import decode_instance_file
from portal_core.load_instance import load_instance
from portal_core.realization import METHOD_AUGMENT, realize
from portal_core.save_instance import save_instance
from portal_core.validate_no_cover import validate_no_cover_block

from Recipe_Selection._execution import cap_process_workers, initialize_numeric_worker

from Recipe_Selection.suite_pipeline.discovery import (
    RecipeRecord,
    read_recipe_record,
)

from .config import serializable_config


SCHEMA_VERSION = 1
_MANAGED_OUTPUT_NAMES = {"instances", "logs", "metadata"}
_ANCHOR_CACHE: dict[str, dict[str, Any]] = {}
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SourceRecord:
    """Validated source-recipe metadata and content identity."""

    recipe: RecipeRecord
    sha256: str
    seed_key: int


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_token(value: str, fallback: str) -> str:
    token = _SAFE_TOKEN.sub("_", str(value).strip()).strip(". _")
    return token or fallback


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _blocks_of(inst: dict[str, Any]) -> list[tuple[int, int, list[int], float]]:
    """Return component slices, global dimension indices, and block weights."""

    dimension = int(inst["d"])
    components = int(inst["K"])
    schema = inst.get("_schema_blocks") or [
        {
            "K": components,
            "d": dimension,
            "dim_indices": list(range(dimension)),
            "weight": 1.0,
        }
    ]
    result: list[tuple[int, int, list[int], float]] = []
    component_start = 0
    dimension_start = 0
    for block in schema:
        block_components = int(block["K"])
        block_dimension = int(block.get("d", dimension))
        indices = block.get("dim_indices")
        if indices is None:
            indices = list(range(dimension_start, dimension_start + block_dimension))
        result.append(
            (
                component_start,
                component_start + block_components,
                [int(value) for value in indices],
                float(block.get("weight", 1.0)),
            )
        )
        component_start += block_components
        dimension_start += block_dimension
    return result


def _verify_augmented_instance(
    inst: dict[str, Any],
    *,
    source_anchor: dict[str, Any],
    center_range: tuple[float, float],
    beta_range: tuple[float, float],
    beta_gap: float,
    beta_opt_gap: float,
    ensure_no_cover: bool,
    no_cover_tolerance: float,
) -> None:
    """Independently verify the saved Type 3 realization contract."""

    realization = (inst.get("_suite", {}) or {}).get("_realization", {}) or {}
    if realization.get("method") != METHOD_AUGMENT:
        raise AssertionError("Saved instance does not declare a Type 3 realization.")

    parameters = inst["P"]
    centers = np.asarray(parameters["c"], dtype=float)
    betas = np.asarray(parameters["beta"], dtype=float)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(betas)):
        raise AssertionError("Saved centers and basin depths must be finite.")

    source_blocks = _blocks_of(source_anchor)
    realized_blocks = _blocks_of(inst)
    if len(realized_blocks) != len(source_blocks):
        raise AssertionError("Type 3 changed the number of Layer-3 blocks.")
    for source_block, realized_block in zip(source_blocks, realized_blocks):
        source_start, source_end, source_indices, _ = source_block
        realized_start, realized_end, realized_indices, _ = realized_block
        if source_end - source_start != realized_end - realized_start:
            raise AssertionError("Type 3 changed a block's component count.")
        if len(source_indices) != len(realized_indices):
            raise AssertionError("Type 3 changed a block's local dimension.")
    source_weights = sorted(float(block[3]) for block in source_blocks)
    realized_weights = sorted(float(block[3]) for block in realized_blocks)
    if not np.all(np.isfinite(realized_weights)) or not np.allclose(
        source_weights,
        realized_weights,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Type 3 did not preserve the source block-weight multiset."
        )

    optimum = np.zeros(int(inst["d"]), dtype=float)
    optimum_value = 0.0

    for start, end, indices, weight in realized_blocks:
        index_array = np.asarray(indices, dtype=int)
        local_centers = centers[start:end, :][:, index_array]
        if local_centers.size:
            if float(local_centers.min()) < center_range[0] - 1e-9:
                raise AssertionError("A component center is below center_min.")
            if float(local_centers.max()) > center_range[1] + 1e-9:
                raise AssertionError("A component center is above center_max.")
        local_betas = betas[start:end]
        if float(local_betas.min()) < beta_range[0] - 1e-9:
            raise AssertionError("A basin depth is below beta_min.")
        if float(local_betas.max()) > beta_range[1] + 1e-9:
            raise AssertionError("A basin depth is above beta_max.")
        local_min = float(local_betas.min())
        if float(local_betas.max()) - local_min > float(beta_gap) + 1e-9:
            raise AssertionError("A block exceeds the requested beta_gap.")
        best_local = int(np.argmin(local_betas))
        if local_betas.size > 1:
            nonoptimal = np.delete(local_betas, best_local)
            if np.any(nonoptimal - local_min < float(beta_opt_gap) - 1e-9):
                raise AssertionError(
                    "A nonoptimal basin violates the requested beta_opt_gap."
                )

        best_component = start + best_local
        optimum[index_array] = centers[best_component, index_array]
        optimum_value += weight * float(np.min(local_betas))

        if ensure_no_cover:
            ok, offenders = validate_no_cover_block(
                inst,
                component_indices=range(start, end),
                dimension_indices=index_array,
                tol=float(no_cover_tolerance),
            )
            if not ok:
                labels = [int(item[0]) + 1 for item in offenders]
                raise AssertionError(
                    f"Covered component centers remain in a block: {labels}."
                )

    evaluated = float(np.squeeze(fitness(optimum, inst)))
    if not math.isfinite(evaluated) or not math.isfinite(optimum_value):
        raise AssertionError("Known-optimum verification produced a nonfinite value.")
    tolerance = 1e-4 + 1e-6 * abs(optimum_value)
    if abs(evaluated - optimum_value) > tolerance:
        raise AssertionError(
            "Known-optimum consistency failed: "
            f"fitness(x*)={evaluated}, expected={optimum_value}."
        )


def _load_anchor_cached(path: str) -> dict[str, Any]:
    anchor = _ANCHOR_CACHE.get(path)
    if anchor is None:
        # Submitted chunks are grouped by source recipe, so one cached anchor
        # captures the useful locality without retaining an unbounded source
        # bank in every long-lived worker process.
        _ANCHOR_CACHE.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            anchor = load_instance(path)
        _ANCHOR_CACHE[path] = anchor
    return anchor


def _generate_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker entry point: generate and atomically save one sample chunk."""

    source_path = str(task["source_path"])
    anchor = _load_anchor_cached(source_path)
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = task["generation"]
    validation = task["validation"]
    rows: list[dict[str, Any]] = []

    for sample_id in task["sample_ids"]:
        started = time.perf_counter()
        final_written = False
        task_id = f"{task['source_sha256']}:{int(sample_id)}"
        output_name = (
            f"{task['recipe_id']}__type3_s{int(sample_id):06d}.json"
        )
        final_path = output_dir / output_name
        temporary_name = (
            f".{output_name[:-5]}.{os.getpid()}.partial.json"
        )
        temporary_path = output_dir / temporary_name
        if temporary_path.exists():
            temporary_path.unlink()

        try:
            inst = realize(
                anchor,
                int(generation["base_seed"]),
                int(sample_id),
                method=METHOD_AUGMENT,
                center_range=(
                    float(generation["center_min"]),
                    float(generation["center_max"]),
                ),
                beta_range=(
                    float(generation["beta_min"]),
                    float(generation["beta_max"]),
                ),
                beta_gap=float(generation["beta_gap"]),
                beta_opt_gap=float(generation["beta_opt_gap"]),
                recipe_seed_key=int(task["recipe_seed_key"]),
                ensure_no_cover=bool(generation["ensure_no_cover"]),
                no_cover_max_attempts=int(generation["no_cover_max_attempts"]),
                no_cover_tolerance=float(generation["no_cover_tolerance"]),
                trace=bool(generation["trace_metadata"]),
            )
            suite = inst.setdefault("_suite", {})
            suite["source_recipe_id"] = str(task["recipe_id"])
            suite["source_recipe_sha256"] = str(task["source_sha256"])
            suite["augmentation_instance_id"] = task_id
            suite["augmentation_run"] = str(task["run_name"])
            suite["augmentation_contract_sha256"] = str(
                task["generation_signature"]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                written = Path(
                    save_instance(
                        inst,
                        temporary_name,
                        output_dir=output_dir,
                    )
                )
            os.replace(written, final_path)
            final_written = True

            verified = False
            if bool(validation["verify_outputs"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    reloaded = load_instance(str(final_path))
                _verify_augmented_instance(
                    reloaded,
                    source_anchor=anchor,
                    center_range=(
                        float(generation["center_min"]),
                        float(generation["center_max"]),
                    ),
                    beta_range=(
                        float(generation["beta_min"]),
                        float(generation["beta_max"]),
                    ),
                    beta_gap=float(generation["beta_gap"]),
                    beta_opt_gap=float(generation["beta_opt_gap"]),
                    ensure_no_cover=bool(generation["ensure_no_cover"]),
                    no_cover_tolerance=float(generation["no_cover_tolerance"]),
                )
                verified = True

            realization = (inst.get("_suite", {}) or {}).get(
                "_realization", {}
            ) or {}
            no_cover = (
                ((realization.get("type3", {}) or {}).get("no_cover", {})) or {}
            )
            rows.append(
                {
                    "task_id": task_id,
                    "source_recipe_id": str(task["recipe_id"]),
                    "family": str(task["family"]),
                    "family_id": task.get("family_id", ""),
                    "dimension": int(task["dimension"]),
                    "block_count": int(task["block_count"]),
                    "sample_id": int(sample_id),
                    "base_seed": int(generation["base_seed"]),
                    "source_sha256": str(task["source_sha256"]),
                    "generation_signature": str(task["generation_signature"]),
                    "output_path": str(final_path),
                    "output_sha256": _hash_file(final_path),
                    "verified": bool(verified),
                    "no_cover_enforced": bool(generation["ensure_no_cover"]),
                    "no_cover_rejections": int(no_cover.get("rejections", 0)),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "completed_at": _utc_now(),
                }
            )
        except Exception as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            if final_written and final_path.exists():
                final_path.unlink()
            raise RuntimeError(
                f"Type 3 generation failed for recipe={task['recipe_id']}, "
                f"sample_id={sample_id}: {exc}"
            ) from exc

    return rows


class AugmentationDatasetRunner:
    """Generate a Type 3 JSON dataset with provenance and safe resumption."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: str | Path,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.output_root = Path(config["run"]["output_root"]).resolve()
        self.dry_run = bool(dry_run)
        self.force = bool(force)
        self.sources: list[SourceRecord] = []
        self.source_bank_fingerprint = ""
        self.implementation_fingerprint = ""
        self.generation_signature = ""
        self.run_signature = ""
        self.prior_elapsed_seconds = 0.0
        self._thread_environment_before: dict[str, str | None] = {}
        self.logger = logging.getLogger(
            f"portal.augmentation.{id(self)}"
        )
        self.paths = {
            "instances": self.output_root / "instances",
            "metadata": self.output_root / "metadata",
            "logs": self.output_root / "logs",
            "log": self.output_root / "logs" / "augmentation.log",
            "state": self.output_root / "metadata" / "run_state.json",
            "checkpoint": self.output_root / "metadata" / "completed_tasks.jsonl",
            "source_manifest": self.output_root / "metadata" / "source_manifest.csv",
            "instance_manifest": (
                self.output_root / "metadata" / "instance_manifest.csv"
            ),
            "resolved_config": self.output_root / "metadata" / "config.resolved.json",
            "run_manifest": self.output_root / "metadata" / "run_manifest.json",
            "summary_json": self.output_root / "metadata" / "summary.json",
            "summary_md": self.output_root / "metadata" / "summary.md",
        }

    def _discover_sources(self) -> None:
        paths: list[Path] = []
        folder_inputs: list[Path] = []
        for raw in self.config["run"]["sources"]:
            source = Path(raw).resolve()
            if source.is_dir():
                folder_inputs.append(source)
                paths.extend(
                    sorted(
                        path for path in source.rglob("*.json") if path.is_file()
                    )
                )
            elif source.is_file() and source.suffix.lower() == ".json":
                paths.append(source)
            else:
                raise FileNotFoundError(
                    f"Source is not a JSON file or directory: {source}"
                )

        for folder in folder_inputs:
            if self.output_root == folder or self.output_root.is_relative_to(folder):
                raise ValueError(
                    "run.output_root must not be inside a source directory; "
                    f"otherwise generated JSONs would become future inputs: {folder}"
                )
        unique_paths = sorted({path.resolve() for path in paths})
        if not unique_paths:
            raise FileNotFoundError("No source recipe JSON files were found.")
        output_root = self.output_root.resolve()
        for path in unique_paths:
            if path == output_root or path.is_relative_to(output_root):
                raise ValueError(
                    "Source recipes must not be read from run.output_root; "
                    f"received {path}. Choose an independent source bank."
                )

        records: list[SourceRecord] = []
        seen_ids: dict[str, Path] = {}
        seen_hashes: dict[str, Path] = {}
        generation = self.config["generation"]
        center_min = float(generation["center_min"])
        center_max = float(generation["center_max"])
        for path in unique_paths:
            recipe = read_recipe_record(path)
            key = recipe.recipe_id.casefold()
            if key in seen_ids:
                raise ValueError(
                    "Duplicate source recipe identifier under case-insensitive "
                    f"semantics: {recipe.recipe_id!r} in {seen_ids[key]} and {path}."
                )
            seen_ids[key] = path
            payload = decode_instance_file(json.loads(path.read_text(encoding="utf-8-sig")))
            with contextlib.redirect_stdout(io.StringIO()):
                runtime = load_instance(str(path))
            if int(runtime["d"]) != recipe.dimension:
                raise ValueError(
                    f"{path}: loaded dimension {runtime['d']} disagrees with "
                    f"metadata dimension {recipe.dimension}."
                )
            lower = np.asarray(payload.get("bounds_min"), dtype=float)
            upper = np.asarray(payload.get("bounds_max"), dtype=float)
            if lower.size == 0 or upper.size == 0:
                raise ValueError(f"{path}: bounds_min and bounds_max are required.")
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError(f"{path}: search bounds must be finite.")
            if center_min < float(np.max(lower)) - 1e-12:
                raise ValueError(
                    f"{path}: configured center_min={center_min} lies below at "
                    f"least one search-domain lower bound."
                )
            if center_max > float(np.min(upper)) + 1e-12:
                raise ValueError(
                    f"{path}: configured center_max={center_max} lies above at "
                    f"least one search-domain upper bound."
                )
            content = path.read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()
            if sha256 in seen_hashes:
                raise ValueError(
                    "Duplicate source-recipe content is not allowed because it "
                    "would duplicate the Type 3 random stream and task identity: "
                    f"{seen_hashes[sha256]} and {path}."
                )
            seen_hashes[sha256] = path
            # Derive the Type 3 recipe key from the source-file bytes for a
            # deterministic recipe-specific stream.
            seed_key = int.from_bytes(
                hashlib.blake2b(content, digest_size=8).digest(),
                "big",
                signed=False,
            )
            records.append(
                SourceRecord(recipe=recipe, sha256=sha256, seed_key=seed_key)
            )

        self.sources = sorted(
            records,
            key=lambda item: (
                (
                    item.recipe.family_id
                    if item.recipe.family_id is not None
                    else 10**9
                ),
                (
                    item.recipe.recipe_index
                    if item.recipe.recipe_index is not None
                    else 10**9
                ),
                item.recipe.recipe_id.casefold(),
            ),
        )
        # Sanitization makes output paths portable, but distinct identifiers
        # such as ``recipe a`` and ``recipe_a`` can collapse to the same token.
        # Reject the selected layout before any worker can overwrite another
        # source recipe's output.
        output_keys: dict[str, SourceRecord] = {}
        first_sample = int(self.config["generation"]["start_sample"])
        for source in self.sources:
            recipe_token = _safe_token(source.recipe.recipe_id, "recipe")
            filename = f"{recipe_token}__type3_s{first_sample:06d}.json"
            candidate = self._output_directory(source) / filename
            key = str(candidate).casefold()
            previous = output_keys.get(key)
            if previous is not None:
                raise ValueError(
                    "Source recipe identifiers collide after portable output-path "
                    f"sanitization under layout={self.config['output']['layout']!r}: "
                    f"{previous.recipe.recipe_id!r} and "
                    f"{source.recipe.recipe_id!r}."
                )
            output_keys[key] = source
        self.source_bank_fingerprint = _hash_payload(
            [
                {
                    "recipe_id": item.recipe.recipe_id,
                    "sha256": item.sha256,
                }
                for item in self.sources
            ]
        )

    def _compute_signatures(self) -> None:
        code_paths = [
            self.project_root / "portal_core" / "realization.py",
            self.project_root / "portal_core" / "benchmark_generator.py",
            self.project_root / "portal_core" / "save_instance.py",
            self.project_root / "portal_core" / "load_instance.py",
            self.project_root / "portal_core" / "fitness.py",
            self.project_root / "portal_core" / "validate_no_cover.py",
            self.project_root / "Recipe_Selection" / "_execution.py",
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("config.py"),
        ]
        self.implementation_fingerprint = _hash_payload(
            {
                str(path.relative_to(self.project_root)): _hash_file(path)
                for path in code_paths
            }
        )
        contract = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.config["run"]["name"],
            "generation": self.config["generation"],
            # Output verification changes the recorded contract. Resume-hash
            # checking is operational only and may be changed between runs.
            "validation": {
                "verify_outputs": self.config["validation"]["verify_outputs"]
            },
            "output": self.config["output"],
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy": np.__version__,
            "implementation_fingerprint": self.implementation_fingerprint,
        }
        self.generation_signature = _hash_payload(contract)
        self.run_signature = _hash_payload(
            {
                "generation_signature": self.generation_signature,
                "source_bank_fingerprint": self.source_bank_fingerprint,
            }
        )

    def _guard_output_root(self) -> None:
        if self.output_root == self.project_root:
            raise ValueError("run.output_root must not be the repository root.")
        if self.project_root.is_relative_to(self.output_root):
            raise ValueError("run.output_root must not contain the repository root.")

    def _clear_managed_output(self) -> None:
        if not self.output_root.exists():
            return
        state = self.paths["state"]
        if any(self.output_root.iterdir()) and not state.is_file():
            raise RuntimeError(
                "Refusing --force because the non-empty output directory has no "
                "PORTAL augmentation run_state.json marker. Choose a new output_root."
            )
        if state.is_file():
            try:
                marker = json.loads(state.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Refusing --force because the augmentation state marker "
                    "cannot be validated."
                ) from exc
            required = (
                marker.get("schema_version") == SCHEMA_VERSION
                and isinstance(marker.get("run_signature"), str)
                and bool(marker.get("run_signature"))
                and isinstance(marker.get("generation_signature"), str)
                and bool(marker.get("generation_signature"))
                and isinstance(marker.get("source_bank_fingerprint"), str)
                and bool(marker.get("source_bank_fingerprint"))
            )
            if not required:
                raise RuntimeError(
                    "Refusing --force because run_state.json is not a valid "
                    "PORTAL augmentation state marker."
                )
        for name in _MANAGED_OUTPUT_NAMES:
            target = (self.output_root / name).resolve()
            if target.parent != self.output_root:
                raise RuntimeError(f"Unsafe managed output path: {target}")
            if target.exists():
                shutil.rmtree(target)

    def _setup_logger(self) -> None:
        self.paths["logs"].mkdir(parents=True, exist_ok=True)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        formatter = _UtcFormatter(
            "%(asctime)sZ | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(self.paths["log"], encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

    def _close_logger(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def _set_worker_thread_environment(self) -> None:
        variables = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if not self._thread_environment_before:
            self._thread_environment_before = {
                variable: os.environ.get(variable) for variable in variables
            }
        for variable in variables:
            os.environ[variable] = "1"

    def _restore_worker_thread_environment(self) -> None:
        for variable, previous in self._thread_environment_before.items():
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        self._thread_environment_before.clear()

    def _initial_state(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.config["run"]["name"],
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "run_signature": self.run_signature,
            "generation_signature": self.generation_signature,
            "source_bank_fingerprint": self.source_bank_fingerprint,
            "source_recipe_count": len(self.sources),
            "expected_instance_count": self.total_tasks,
            "completed_instance_count": 0,
        }

    def _prepare_output(self) -> dict[str, Any]:
        self._guard_output_root()
        if self.force:
            self._clear_managed_output()
        self.output_root.mkdir(parents=True, exist_ok=True)

        state_path = self.paths["state"]
        existing: dict[str, Any] | None = None
        if state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("run_signature") != self.run_signature:
                raise RuntimeError(
                    "The output directory belongs to an incompatible augmentation "
                    "contract or source bank. Use a new output_root or rerun with "
                    "--force to replace the managed run artifacts."
                )
            if not bool(self.config["run"]["resume"]):
                raise RuntimeError(
                    "Compatible prior output exists but run.resume=false. Use "
                    "--force for a clean regeneration or enable resume."
                )
            self.prior_elapsed_seconds = float(
                existing.get(
                    "cumulative_elapsed_seconds",
                    existing.get("elapsed_seconds", 0.0),
                )
            )
        elif any(self.output_root.iterdir()):
            raise RuntimeError(
                "The output directory is non-empty but has no compatible PORTAL "
                "augmentation state. Choose a new output_root."
            )

        for key in ("instances", "metadata", "logs"):
            self.paths[key].mkdir(parents=True, exist_ok=True)
        self._setup_logger()
        state = existing or self._initial_state()
        state["status"] = "running"
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)
        return state

    @property
    def total_tasks(self) -> int:
        return len(self.sources) * int(self.config["generation"]["count_per_recipe"])

    def _output_directory(self, source: SourceRecord) -> Path:
        layout = self.config["output"]["layout"]
        family = _safe_token(source.recipe.family, "UnknownFamily")
        recipe = _safe_token(source.recipe.recipe_id, "recipe")
        if layout == "flat":
            return self.paths["instances"]
        if layout == "by_recipe":
            return self.paths["instances"] / recipe
        return self.paths["instances"] / family / recipe

    def _task_id(self, source: SourceRecord, sample_id: int) -> str:
        return f"{source.sha256}:{int(sample_id)}"

    def _source_rows(self) -> list[dict[str, Any]]:
        count = int(self.config["generation"]["count_per_recipe"])
        return [
            {
                "recipe_id": item.recipe.recipe_id,
                "family": item.recipe.family,
                "family_id": (
                    item.recipe.family_id
                    if item.recipe.family_id is not None
                    else ""
                ),
                "recipe_index": (
                    item.recipe.recipe_index
                    if item.recipe.recipe_index is not None
                    else ""
                ),
                "dimension": item.recipe.dimension,
                "block_count": item.recipe.block_count,
                "source_sha256": item.sha256,
                "instances_requested": count,
                "source_path": _portable_path(item.recipe.path, self.project_root),
            }
            for item in self.sources
        ]

    def _write_provenance(self) -> None:
        source_fields = [
            "recipe_id",
            "family",
            "family_id",
            "recipe_index",
            "dimension",
            "block_count",
            "source_sha256",
            "instances_requested",
            "source_path",
        ]
        _write_csv_atomic(
            self.paths["source_manifest"], self._source_rows(), source_fields
        )
        _write_json_atomic(
            self.paths["resolved_config"], serializable_config(self.config)
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "run_name": self.config["run"]["name"],
            "purpose": (
                "Type 3 large-scale dataset augmentation from a source recipe bank"
            ),
            "scope_note": (
                "No ELA, Performance, optimizer evaluation, deduplication, or suite "
                "selection is performed. Dataset scope is inherited from the "
                "supplied recipe bank."
            ),
            "project_root": str(self.project_root),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "git_commit": _git_commit(self.project_root),
            "source_recipe_count": len(self.sources),
            "expected_instance_count": self.total_tasks,
            "source_bank_fingerprint": self.source_bank_fingerprint,
            "implementation_fingerprint": self.implementation_fingerprint,
            "generation_signature": self.generation_signature,
            "run_signature": self.run_signature,
            "config": serializable_config(self.config),
        }
        _write_json_atomic(self.paths["run_manifest"], manifest)

    def _load_checkpoint(self) -> dict[str, dict[str, Any]]:
        path = self.paths["checkpoint"]
        if not path.is_file() or not bool(self.config["run"]["resume"]):
            return {}
        start = int(self.config["generation"]["start_sample"])
        count = int(self.config["generation"]["count_per_recipe"])
        end = start + count
        source_by_hash = {source.sha256: source for source in self.sources}

        candidates: dict[str, dict[str, Any]] = {}
        malformed = 0
        unexpected = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    self.logger.warning(
                        "Ignoring malformed checkpoint line %d.", line_number
                    )
                    continue
                task_id = str(row.get("task_id", ""))
                try:
                    source_hash, sample_text = task_id.rsplit(":", 1)
                    sample_id = int(sample_text)
                except (ValueError, TypeError):
                    unexpected += 1
                    continue
                if source_hash not in source_by_hash or not start <= sample_id < end:
                    unexpected += 1
                    continue
                candidates[task_id] = row

        valid: dict[str, dict[str, Any]] = {}
        verify_hashes = bool(self.config["validation"]["verify_hashes_on_resume"])
        for task_id, row in candidates.items():
            source_hash = task_id.rsplit(":", 1)[0]
            source = source_by_hash[source_hash]
            if row.get("generation_signature") != self.generation_signature:
                continue
            if row.get("source_sha256") != source.sha256:
                continue
            output = Path(str(row.get("output_path", "")))
            if not output.is_absolute():
                output = self.output_root / output
            output = output.resolve()
            if not output.is_relative_to(self.paths["instances"].resolve()):
                continue
            if not output.is_file():
                continue
            if verify_hashes and _hash_file(output) != row.get("output_sha256"):
                self.logger.warning(
                    "Regenerating %s because its saved hash does not match.", task_id
                )
                continue
            normalized = dict(row)
            normalized["output_path"] = str(output)
            valid[task_id] = normalized
        self.logger.info(
            "Resume audit: %d valid task(s), %d stale/invalid task(s), "
            "%d unexpected task row(s), %d malformed line(s).",
            len(valid),
            len(candidates) - len(valid),
            unexpected,
            malformed,
        )
        return valid

    def _iter_pending_chunks(
        self, completed: dict[str, dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        generation = self.config["generation"]
        start = int(generation["start_sample"])
        count = int(generation["count_per_recipe"])
        chunk_size = int(self.config["execution"]["task_chunk_size"])
        for source in self.sources:
            pending: list[int] = []
            for sample_id in range(start, start + count):
                if self._task_id(source, sample_id) in completed:
                    continue
                pending.append(sample_id)
                if len(pending) == chunk_size:
                    yield self._chunk_payload(source, pending)
                    pending = []
            if pending:
                yield self._chunk_payload(source, pending)

    def _chunk_payload(
        self, source: SourceRecord, sample_ids: Iterable[int]
    ) -> dict[str, Any]:
        return {
            "source_path": str(source.recipe.path),
            "source_sha256": source.sha256,
            "recipe_seed_key": source.seed_key,
            "recipe_id": source.recipe.recipe_id,
            "family": source.recipe.family,
            "family_id": source.recipe.family_id,
            "dimension": source.recipe.dimension,
            "block_count": source.recipe.block_count,
            "sample_ids": list(sample_ids),
            "output_dir": str(self._output_directory(source)),
            "run_name": self.config["run"]["name"],
            "generation_signature": self.generation_signature,
            "generation": self.config["generation"],
            "validation": self.config["validation"],
        }

    def _pending_chunk_count(
        self, completed: dict[str, dict[str, Any]]
    ) -> int:
        completed_by_hash: dict[str, int] = {}
        for row in completed.values():
            source_hash = str(row["source_sha256"])
            completed_by_hash[source_hash] = completed_by_hash.get(source_hash, 0) + 1
        target = int(self.config["generation"]["count_per_recipe"])
        chunk_size = int(self.config["execution"]["task_chunk_size"])
        return sum(
            math.ceil(
                max(0, target - completed_by_hash.get(source.sha256, 0))
                / chunk_size
            )
            for source in self.sources
        )

    def _resolve_workers(self, pending_chunks: int) -> int:
        cpus = max(1, os.cpu_count() or 1)
        configured = int(self.config["execution"]["max_workers"])
        if configured > 0:
            workers = min(configured, cpus)
        else:
            workers = max(
                1,
                int(math.floor(cpus * float(self.config["execution"]["cpu_fraction"]))),
            )
        return cap_process_workers(workers, cpus, task_count=pending_chunks)

    def _update_state(
        self,
        state: dict[str, Any],
        *,
        status: str,
        completed: int,
        started: float,
        error: str | None = None,
    ) -> None:
        state["status"] = status
        state["updated_at"] = _utc_now()
        state["completed_instance_count"] = int(completed)
        invocation_elapsed = time.monotonic() - started
        cumulative_elapsed = self.prior_elapsed_seconds + invocation_elapsed
        state["invocation_elapsed_seconds"] = round(invocation_elapsed, 3)
        state["cumulative_elapsed_seconds"] = round(cumulative_elapsed, 3)
        state["elapsed_seconds"] = round(cumulative_elapsed, 3)
        if error is None:
            state.pop("error", None)
        else:
            state["error"] = error
        _write_json_atomic(self.paths["state"], state)

    def _run_parallel(
        self,
        completed: dict[str, dict[str, Any]],
        state: dict[str, Any],
        started: float,
    ) -> dict[str, dict[str, Any]]:
        pending_count = self.total_tasks - len(completed)
        if pending_count <= 0:
            self.logger.info("All %d tasks are already complete.", self.total_tasks)
            return completed

        pending_chunks = self._pending_chunk_count(completed)
        workers = self._resolve_workers(pending_chunks)
        execution = self.config["execution"]
        max_in_flight = max(
            workers,
            workers * int(execution["max_in_flight_factor"]),
        )
        self._set_worker_thread_environment()

        self.logger.info(
            "Execution: workers=%d, chunk_size=%d, max_in_flight_chunks=%d, "
            "pending_instances=%d.",
            workers,
            int(execution["task_chunk_size"]),
            max_in_flight,
            pending_count,
        )
        self.logger.info("Pending worker chunks: %d.", pending_chunks)
        iterator = iter(self._iter_pending_chunks(completed))
        checkpoint_every = int(execution["checkpoint_every"])
        heartbeat_percent = float(execution["heartbeat_percent"])
        alive_seconds = float(execution["heartbeat_alive_seconds"])
        next_percent = (
            math.floor((100.0 * len(completed) / self.total_tasks) / heartbeat_percent)
            + 1
        ) * heartbeat_percent
        last_completion = time.monotonic()
        newly_completed = 0
        since_flush = 0

        checkpoint = self.paths["checkpoint"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        source_completed: dict[str, int] = {}
        for row in completed.values():
            recipe_id = str(row.get("source_recipe_id", ""))
            source_completed[recipe_id] = source_completed.get(recipe_id, 0) + 1
        per_source_target = int(self.config["generation"]["count_per_recipe"])
        with checkpoint.open("a", encoding="utf-8", buffering=1) as checkpoint_handle:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize_numeric_worker,
            ) as executor:
                futures: dict[Future[list[dict[str, Any]]], dict[str, Any]] = {}

                def submit_until_full() -> None:
                    while len(futures) < max_in_flight:
                        try:
                            task = next(iterator)
                        except StopIteration:
                            break
                        futures[executor.submit(_generate_chunk, task)] = task

                submit_until_full()
                while futures:
                    done, _ = wait(
                        futures,
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )
                    now = time.monotonic()
                    if not done:
                        if (
                            alive_seconds > 0.0
                            and now - last_completion >= alive_seconds
                        ):
                            percent = 100.0 * len(completed) / self.total_tasks
                            elapsed = now - started
                            self.logger.info(
                                "Heartbeat alive: %.1f%% (%d/%d), elapsed=%s, "
                                "in_flight_chunks=%d.",
                                percent,
                                len(completed),
                                self.total_tasks,
                                self._format_duration(elapsed),
                                len(futures),
                            )
                            last_completion = now
                        continue

                    for future in done:
                        task = futures.pop(future)
                        try:
                            rows = future.result()
                        except Exception:
                            checkpoint_handle.flush()
                            os.fsync(checkpoint_handle.fileno())
                            self._update_state(
                                state,
                                status="running",
                                completed=len(completed),
                                started=started,
                            )
                            for pending in futures:
                                pending.cancel()
                            self.logger.error(
                                "Worker chunk failed for recipe=%s, sample_ids=%s.",
                                task["recipe_id"],
                                task["sample_ids"],
                            )
                            raise
                        for row in rows:
                            portable = dict(row)
                            portable["output_path"] = _portable_path(
                                Path(row["output_path"]), self.output_root
                            )
                            checkpoint_handle.write(
                                json.dumps(portable, sort_keys=True) + "\n"
                            )
                            completed[row["task_id"]] = row
                            recipe_id = str(row["source_recipe_id"])
                            source_completed[recipe_id] = (
                                source_completed.get(recipe_id, 0) + 1
                            )
                            if source_completed[recipe_id] == per_source_target:
                                self.logger.info(
                                    "Source complete: %s (%d/%d instances).",
                                    recipe_id,
                                    per_source_target,
                                    per_source_target,
                                )
                            newly_completed += 1
                            since_flush += 1
                        last_completion = now

                        if since_flush >= checkpoint_every:
                            checkpoint_handle.flush()
                            os.fsync(checkpoint_handle.fileno())
                            since_flush = 0
                            self._update_state(
                                state,
                                status="running",
                                completed=len(completed),
                                started=started,
                            )

                        percent = 100.0 * len(completed) / self.total_tasks
                        if (
                            percent + 1e-12 >= next_percent
                            or len(completed) == self.total_tasks
                        ):
                            elapsed = max(now - started, 1e-9)
                            rate = newly_completed / elapsed
                            remaining = self.total_tasks - len(completed)
                            eta = remaining / rate if rate > 0.0 else math.inf
                            self.logger.info(
                                "Progress: %.1f%% (%d/%d), elapsed=%s, ETA=%s, "
                                "throughput=%.2f instances/s.",
                                percent,
                                len(completed),
                                self.total_tasks,
                                self._format_duration(elapsed),
                                self._format_duration(eta),
                                rate,
                            )
                            while next_percent <= percent + 1e-12:
                                next_percent += heartbeat_percent
                        submit_until_full()

            checkpoint_handle.flush()
            os.fsync(checkpoint_handle.fileno())
        self._update_state(
            state,
            status="running",
            completed=len(completed),
            started=started,
        )
        return completed

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not math.isfinite(seconds):
            return "unknown"
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes:d}m {seconds:02d}s"
        return f"{seconds:d}s"

    def _final_rows(
        self, completed: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        source_order = {
            item.sha256: index for index, item in enumerate(self.sources)
        }
        rows: list[dict[str, Any]] = []
        for row in completed.values():
            normalized = dict(row)
            output = Path(str(normalized["output_path"]))
            if output.is_absolute():
                normalized["output_path"] = _portable_path(output, self.output_root)
            rows.append(normalized)
        rows.sort(
            key=lambda row: (
                source_order.get(str(row["task_id"]).split(":", 1)[0], 10**9),
                int(row["sample_id"]),
            )
        )
        return rows

    def _write_final_outputs(
        self,
        completed: dict[str, dict[str, Any]],
        *,
        elapsed_seconds: float,
    ) -> None:
        rows = self._final_rows(completed)
        fields = [
            "task_id",
            "source_recipe_id",
            "family",
            "family_id",
            "dimension",
            "block_count",
            "sample_id",
            "base_seed",
            "source_sha256",
            "generation_signature",
            "output_path",
            "output_sha256",
            "verified",
            "no_cover_enforced",
            "no_cover_rejections",
            "elapsed_seconds",
            "completed_at",
        ]
        _write_csv_atomic(self.paths["instance_manifest"], rows, fields)

        family_sources: dict[str, int] = {}
        for source in self.sources:
            family_sources[source.recipe.family] = (
                family_sources.get(source.recipe.family, 0) + 1
            )
        family_instances: dict[str, int] = {}
        for row in rows:
            family = str(row["family"])
            family_instances[family] = family_instances.get(family, 0) + 1
        summary = {
            "completed_at": _utc_now(),
            "run_name": self.config["run"]["name"],
            "source_recipe_count": len(self.sources),
            "generated_instance_count": len(rows),
            "count_per_recipe": int(self.config["generation"]["count_per_recipe"]),
            "family_source_counts": family_sources,
            "family_instance_counts": family_instances,
            "verified_instance_count": sum(bool(row["verified"]) for row in rows),
            "total_no_cover_rejections": sum(
                int(row["no_cover_rejections"]) for row in rows
            ),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "scope_note": (
                "Instances were generated only by Type 3 augmentation. No ELA, Performance, "
                "optimizer evaluation, deduplication, or subset selection was run."
            ),
        }
        _write_json_atomic(self.paths["summary_json"], summary)
        family_lines = "\n".join(
            f"- {family}: {family_sources[family]} source recipe(s), "
            f"{family_instances.get(family, 0)} generated instance(s)"
            for family in sorted(family_sources)
        )
        self.paths["summary_md"].write_text(
            "# PORTAL Type 3 Augmentation Summary\n\n"
            f"- Run: `{self.config['run']['name']}`\n"
            f"- Source recipes: {len(self.sources):,}\n"
            "- Instances per recipe: "
            f"{int(self.config['generation']['count_per_recipe']):,}\n"
            f"- Generated instances: {len(rows):,}\n"
            "- Independently verified instances: "
            f"{summary['verified_instance_count']:,}\n"
            "- Rejected center/depth proposals: "
            f"{summary['total_no_cover_rejections']:,}\n"
            f"- Elapsed time: {self._format_duration(elapsed_seconds)}\n\n"
            "## Family Composition\n\n"
            f"{family_lines}\n\n"
            "## Scope\n\n"
            "This workflow materializes canonical unified JSON instances using "
            "Type 3 augmentation from the supplied source recipe bank. It does not "
            "compute ELA or Performance features, run optimization algorithms, remove "
            "duplicates, or select a new suite. Consequently, the structural scope "
            "of the dataset is determined by the supplied source bank and the "
            "declared Type 3 generation ranges.\n",
            encoding="utf-8",
        )

    def _print_plan(self) -> None:
        family_counts: dict[str, int] = {}
        dimensions: dict[int, int] = {}
        for source in self.sources:
            family_counts[source.recipe.family] = (
                family_counts.get(source.recipe.family, 0) + 1
            )
            dimensions[source.recipe.dimension] = (
                dimensions.get(source.recipe.dimension, 0) + 1
            )
        pending_chunks = len(self.sources) * math.ceil(
            int(self.config["generation"]["count_per_recipe"])
            / int(self.config["execution"]["task_chunk_size"])
        )
        print("=" * 72)
        print("PORTAL Type 3 large-scale dataset augmentation")
        print("=" * 72)
        print(f"Run:              {self.config['run']['name']}")
        print(f"Source recipes:   {len(self.sources)}")
        print(f"Families:         {family_counts}")
        print(f"Dimensions:       {dimensions}")
        per_recipe = int(self.config["generation"]["count_per_recipe"])
        print(
            "Requested output: "
            f"{len(self.sources)} x {per_recipe} = {self.total_tasks} instances"
        )
        print(f"Full-plan chunks: {pending_chunks}")
        configured_workers = int(self.config["execution"]["max_workers"])
        if configured_workers > 0:
            worker_text = str(configured_workers)
        else:
            worker_text = (
                "automatic "
                f"({100.0 * float(self.config['execution']['cpu_fraction']):.0f}% CPU)"
            )
        print(f"Workers:          {worker_text}")
        print(f"Output:           {self.output_root}")
        print("Operations:       Type 3 JSON generation only (no ELA/Performance)")
        if self.dry_run:
            print("Mode:             dry run (no files will be written)")

    def run(self) -> None:
        self._guard_output_root()
        self._discover_sources()
        self._compute_signatures()
        self._print_plan()
        if self.dry_run:
            return

        state = self._prepare_output()
        started = time.monotonic()
        try:
            self._write_provenance()
            self.logger.info(
                "Run started: name=%s, sources=%d, expected_instances=%d, "
                "run_signature=%s.",
                self.config["run"]["name"],
                len(self.sources),
                self.total_tasks,
                self.run_signature,
            )
            completed = self._load_checkpoint()
            self._update_state(
                state,
                status="running",
                completed=len(completed),
                started=started,
            )
            completed = self._run_parallel(completed, state, started)
            if len(completed) != self.total_tasks:
                raise RuntimeError(
                    f"Run ended with {len(completed)} completed tasks; "
                    f"expected {self.total_tasks}."
                )
            invocation_elapsed = time.monotonic() - started
            cumulative_elapsed = self.prior_elapsed_seconds + invocation_elapsed
            self._write_final_outputs(
                completed, elapsed_seconds=cumulative_elapsed
            )
            self._update_state(
                state,
                status="completed",
                completed=len(completed),
                started=started,
            )
            self.logger.info(
                "Run completed: %d instances in %s. Manifest: %s",
                len(completed),
                self._format_duration(cumulative_elapsed),
                self.paths["instance_manifest"],
            )
        except KeyboardInterrupt:
            self._update_state(
                state,
                status="interrupted",
                completed=int(state.get("completed_instance_count", 0)),
                started=started,
                error="Interrupted by user",
            )
            self.logger.warning("Run interrupted; completed tasks remain resumable.")
            raise
        except Exception as exc:
            self._update_state(
                state,
                status="failed",
                completed=int(state.get("completed_instance_count", 0)),
                started=started,
                error=str(exc),
            )
            self.logger.exception("Run failed: %s", exc)
            raise
        finally:
            self._restore_worker_thread_environment()
            self._close_logger()
