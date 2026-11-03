"""Discover and validate canonical unified PORTAL recipe JSON files.

Directory names are deliberately ignored. Recipe identity, family membership,
dimension, and block count are read from each unified JSON payload, so callers
may use a flat directory or any convenient nested layout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_FAMILY_DIGITS = re.compile(r"(\d+)")
_RECIPE_INDEX = re.compile(r"(?:^|_)r(\d+)$", re.IGNORECASE)
_WINDOWS_FORBIDDEN_NAME_CHARS = set('<>:"/\\|?*')


@dataclass(frozen=True)
class RecipeRecord:
    """Validated metadata for one canonical unified PORTAL recipe."""

    path: Path
    recipe_id: str
    family: str
    family_id: int | None
    recipe_index: int | None
    dimension: int
    block_count: int

    def as_row(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "family": self.family,
            "family_id": self.family_id,
            "recipe_index": self.recipe_index,
            "dimension": self.dimension,
            "block_count": self.block_count,
            "json_path": str(self.path),
        }


def _as_positive_int(value: Any, label: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: {label} must be an integer; received {value!r}."
        ) from exc
    if parsed < 1:
        raise ValueError(f"{path}: {label} must be positive; received {parsed}.")
    return parsed


def _family_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"Family{int(value)}"
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"Family{int(text)}"
    return text


def _family_id(value: Any, label: str | None) -> int | None:
    if value is not None and value != "":
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"family_id must be an integer; received {value!r}."
            ) from exc
        if parsed < 1:
            raise ValueError(f"family_id must be positive; received {parsed}.")
        return parsed
    if label:
        match = _FAMILY_DIGITS.search(label)
        if match:
            return int(match.group(1))
    return None


def _normalise_recipe_id(
    payload: dict[str, Any], suite: dict[str, Any], path: Path, family_id: int | None
) -> str:
    candidates = [
        payload.get("recipe"),
        payload.get("recipe_id"),
        suite.get("recipe_id"),
    ]
    instance_file = suite.get("instance_file")
    if instance_file:
        candidates.append(Path(str(instance_file)).stem)
    candidates.append(path.stem)

    value = next(
        (str(item).strip() for item in candidates if item not in (None, "")), ""
    )
    if not value:
        raise ValueError(
            f"{path}: no recipe identifier is available in the JSON or filename."
        )
    if re.fullmatch(r"r\d+", value, flags=re.IGNORECASE) and family_id is not None:
        value = f"f{family_id:02d}_{value.lower()}"
    if (
        value in {".", ".."}
        or any(character in _WINDOWS_FORBIDDEN_NAME_CHARS for character in value)
        or any(ord(character) < 32 for character in value)
        or value.endswith((".", " "))
    ):
        raise ValueError(
            f"{path}: recipe identifier {value!r} is not safe as a portable filename."
        )
    return value


def _recipe_index(value: Any, recipe_id: str) -> int | None:
    if value not in (None, ""):
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"recipe_index must be positive; received {parsed}.")
        return parsed
    match = _RECIPE_INDEX.search(recipe_id)
    return int(match.group(1)) if match else None


def read_recipe_record(path: str | Path) -> RecipeRecord:
    """Read and validate the metadata required by the curation pipeline."""

    from portal_core.instance_format import decode_instance_file

    source = Path(path).resolve()
    try:
        payload = decode_instance_file(json.loads(source.read_text(encoding="utf-8-sig")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read unified PORTAL JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: the JSON root must be an object.")
    if bool(payload.get("composite", False)):
        raise ValueError(
            f"{source}: legacy composite envelopes are not accepted; "
            "use the canonical unified JSON schema."
        )

    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(
            f"{source}: canonical unified JSON requires a non-empty blocks array."
        )
    declared_blocks = _as_positive_int(payload.get("B", len(blocks)), "B", source)
    if declared_blocks != len(blocks):
        raise ValueError(
            f"{source}: B={declared_blocks} disagrees with len(blocks)={len(blocks)}."
        )

    suite_raw = payload.get("_suite", {})
    suite = suite_raw if isinstance(suite_raw, dict) else {}
    top_family = _family_label(payload.get("family"))
    suite_family = _family_label(suite.get("family"))
    if top_family and suite_family and top_family != suite_family:
        raise ValueError(
            f"{source}: top-level family {top_family!r} disagrees with "
            f"_suite.family {suite_family!r}."
        )
    family = suite_family or top_family
    if not family:
        raise ValueError(f"{source}: family metadata is required in the unified JSON.")

    try:
        family_id = _family_id(suite.get("family_id", payload.get("family_id")), family)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    family_match = _FAMILY_DIGITS.search(family)
    if (
        family_match
        and family_id is not None
        and int(family_match.group(1)) != family_id
    ):
        raise ValueError(
            f"{source}: family label {family!r} disagrees with family_id={family_id}."
        )
    recipe_id = _normalise_recipe_id(payload, suite, source, family_id)
    try:
        recipe_index = _recipe_index(
            suite.get("recipe_index", payload.get("recipe_index")), recipe_id
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid recipe_index: {exc}") from exc

    dimension_value = payload.get("d", suite.get("d", payload.get("dimension")))
    dimension = _as_positive_int(dimension_value, "dimension (d)", source)
    return RecipeRecord(
        path=source,
        recipe_id=recipe_id,
        family=family,
        family_id=family_id,
        recipe_index=recipe_index,
        dimension=dimension,
        block_count=declared_blocks,
    )


def _sort_key(record: RecipeRecord) -> tuple[Any, ...]:
    family_key: tuple[int, Any]
    if record.family_id is None:
        family_key = (1, record.family.casefold())
    else:
        family_key = (0, record.family_id)
    recipe_key = record.recipe_index if record.recipe_index is not None else 10**12
    return (*family_key, recipe_key, record.recipe_id.casefold())


def discover_recipes(
    root: str | Path,
    *,
    require_single_dimension: bool = True,
) -> list[RecipeRecord]:
    """Recursively discover unified recipe JSONs without using folder names.

    The designated input root is expected to contain recipe JSONs only. Any
    malformed or non-recipe JSON is reported as an error rather than silently
    skipped, preventing an incomplete candidate pool from entering curation.
    """

    source_root = Path(root).resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Recipe root is not a directory: {source_root}")
    paths = sorted(path for path in source_root.rglob("*.json") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No JSON files found below recipe root: {source_root}")

    records = [read_recipe_record(path) for path in paths]
    by_id: dict[str, tuple[str, Path]] = {}
    for record in records:
        normalized_id = record.recipe_id.casefold()
        if normalized_id in by_id:
            previous_id, previous_path = by_id[normalized_id]
            raise ValueError(
                f"Duplicate recipe_id under case-insensitive file semantics "
                f"({previous_id!r}, {record.recipe_id!r}): "
                f"{previous_path} and {record.path}"
            )
        by_id[normalized_id] = (record.recipe_id, record.path)

    dimensions = sorted({record.dimension for record in records})
    if require_single_dimension and len(dimensions) != 1:
        raise ValueError(
            "The current ELA/Performance curation run requires one common dimension; "
            f"found dimensions {dimensions}. Run dimensions separately."
        )
    return sorted(records, key=_sort_key)


def recipe_fingerprint(records: Iterable[RecipeRecord]) -> str:
    """Return a content fingerprint independent of the directory layout."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.recipe_id.casefold()):
        digest.update(record.recipe_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(record.path.read_bytes()).digest())
    return digest.hexdigest()


def write_recipe_manifest(records: Iterable[RecipeRecord], path: str | Path) -> Path:
    """Write the validated recipe index used by a pipeline run."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.as_row() for record in records]
    fieldnames = [
        "recipe_id",
        "family",
        "family_id",
        "recipe_index",
        "dimension",
        "block_count",
        "json_path",
    ]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination
