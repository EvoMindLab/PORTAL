"""Synchronize a selected-recipe CSV with a directory of unified JSON files."""
from __future__ import annotations

# Read-only compatibility for historical characterization artifacts.
try:
    from Recipe_Selection._terminology import read_characterization_csv
except ModuleNotFoundError as _terminology_error:
    if _terminology_error.name != 'Recipe_Selection':
        raise
    import sys as _source_sys
    from pathlib import Path as _SourcePath
    _source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[1]))
    from Recipe_Selection._terminology import read_characterization_csv

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
from portal_core.instance_format import decode_instance_file
try:
    from .suite_pipeline.discovery import discover_recipes
except ImportError:  # Direct execution from Recipe_Selection/.
    from suite_pipeline.discovery import discover_recipes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=(ROOT / "manifests" / "s50.csv" if (ROOT / "manifests" / "s50.csv").is_file()
                 else ROOT / "selection_output" / "benchmark50_selected.csv"),
    )
    parser.add_argument("--suite-root", type=Path, default=ROOT / "Suite")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "Benchmark_S50")
    args = parser.parse_args()

    selected = read_characterization_csv(args.selection_csv)
    if "recipe_id" not in selected.columns:
        raise ValueError("Selection CSV must contain recipe_id.")
    recipe_ids = selected["recipe_id"].astype(str).tolist()
    if not recipe_ids or len(recipe_ids) != len(set(recipe_ids)):
        raise ValueError("Selection CSV is empty or contains duplicate recipe IDs.")

    records = discover_recipes(args.suite_root, require_single_dimension=True)
    source_by_id = {record.recipe_id: record.path for record in records}
    missing = sorted(set(recipe_ids) - set(source_by_id))
    if missing:
        raise ValueError(f"Selected recipes have no source JSON: {missing}")

    source_root = args.suite_root.resolve()
    target = args.target_dir.resolve()
    if target == source_root or target.is_relative_to(source_root):
        raise ValueError(
            "target-dir must be outside suite-root so materialized outputs cannot "
            "be rediscovered as candidate recipes."
        )
    target.mkdir(parents=True, exist_ok=True)
    selected_names = {f"{recipe_id}.json" for recipe_id in recipe_ids}

    for recipe_id in recipe_ids:
        source = source_by_id[recipe_id]
        payload = decode_instance_file(json.loads(source.read_text(encoding="utf-8-sig")))
        if bool(payload.get("composite", False)) or not isinstance(payload.get("blocks"), list):
            raise ValueError(f"{source} is not a canonical unified PORTAL JSON.")
        destination = target / f"{recipe_id}.json"
        temporary = destination.with_suffix(".json.sync-tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    stale = [path for path in target.glob("*.json") if path.name not in selected_names]
    for path in stale:
        path.unlink()

    actual = {path.name for path in target.glob("*.json")}
    if actual != selected_names:
        raise RuntimeError("Target suite does not exactly match the selection CSV.")

    print(
        f"Synchronized {len(recipe_ids)} unified JSON files in {target}; "
        f"removed {len(stale)} stale files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
