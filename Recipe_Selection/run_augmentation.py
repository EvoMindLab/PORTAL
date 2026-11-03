"""
PORTAL Type 3 Large-Scale Dataset Augmentation
==============================================

Generates a large set of augmented problem instances from a bank of anchor
recipes, using the Type 3 (``augment``) realization mode:

  * every component center is independently resampled, uniformly, from a
    user-supplied center range (per block-local dimension);
  * basin depths follow the frozen-catalog rule: one shared base per block is
    drawn from the user-supplied beta range and per-component offsets are
    bounded by ``--beta-gap`` (default 20), with all non-optimal components
    at least ``--beta-opt-gap`` (default 1) above the block optimum, keeping
    within-block depth gaps compatible with the no-cover safeguard and decoy
    basins away from the optimum;
  * all remaining diversification comes from the Type 2 machinery: the global
    coordinate permutation for multi-block recipes, the signed per-dimension
    shuffles of kappa/p/transform fields, and the rho recomputation.

Type 3 is NOT a difficulty-preserving realization: resampling centers and
betas changes the multimodal arrangement and basin-depth structure, producing
different problems within each recipe's structural family.  It is intended
for building large meta-algorithmic training datasets from a curated recipe
bank, where each recipe acts as a structural template and per-template
redundancy is acceptable.

Seeding is systematic and deterministic: each source recipe receives a stable
content fingerprint, and instance j uses the key
(base_seed, recipe_fingerprint, sample_id=j). The entire dataset is therefore
reproducible from the recipe bank plus (base_seed, count).

Usage
-----
python run_augmentation.py --inputs <folder-or-json> [<more> ...]
                           --out-dir <dir> --count 100 --base-seed 42
                           [--center-min -80 --center-max 80]
                           [--beta-min -1000 --beta-max 1000]
                           [--start-sample 1] [--verify]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (str(ROOT), str(ROOT / "portal_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from portal_core.load_instance import load_instance          # noqa: E402
from portal_core.save_instance import save_instance          # noqa: E402
from portal_core.fitness import fitness                      # noqa: E402
from portal_core.realization import (                        # noqa: E402
    realize,
    METHOD_AUGMENT,
    CENTER_MIN,
    CENTER_MAX,
    BETA_MIN,
    BETA_MAX,
    DEFAULT_BETA_GAP,
    DEFAULT_BETA_OPT_GAP,
)


def collect_input_files(inputs: list[str]) -> list[Path]:
    """Expand files/folders into a sorted, de-duplicated list of recipe JSONs."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        else:
            raise FileNotFoundError(f"Input not found or not a JSON file/folder: {item}")
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    if not unique:
        raise ValueError("No recipe JSON files found in the given inputs.")
    return unique


def recipe_fingerprint(path: Path) -> tuple[int, str, str]:
    """Return a content seed key, content token, and source-path token."""
    digest = hashlib.blake2b(path.read_bytes(), digest_size=8).digest()
    token = digest.hex()
    source_token = hashlib.blake2b(
        str(path.resolve()).encode("utf-8"), digest_size=4
    ).hexdigest()
    return int.from_bytes(digest, byteorder="big", signed=False), token, source_token


def _blocks_of(inst) -> list[tuple[int, int, list[int], float]]:
    """(comp_start, comp_end, dim_idx, weight) per block from _schema_blocks."""
    d = int(inst["d"])
    K = int(inst["K"])
    schema = inst.get("_schema_blocks") or [
        {"K": K, "d": d, "dim_indices": list(range(d)), "weight": 1.0}
    ]
    out = []
    ck = cd = 0
    for meta in schema:
        kb = int(meta["K"])
        bd = int(meta.get("d", d))
        di = meta.get("dim_indices")
        if di is None:
            di = list(range(cd, cd + bd))
        out.append((ck, ck + kb, [int(x) for x in di], float(meta.get("weight", 1.0))))
        ck += kb
        cd += bd
    return out


def verify_instance(inst,
                    center_range: tuple[float, float],
                    beta_range: tuple[float, float]) -> None:
    """Raise AssertionError if the augmented instance violates its contract."""
    P = inst["P"]
    c = np.asarray(P["c"], dtype=float)
    beta = np.asarray(P["beta"], dtype=float)
    d = int(inst["d"])

    x_opt = np.zeros(d)
    f_star = 0.0
    for (s, e, di, w) in _blocks_of(inst):
        di_arr = np.asarray(di, dtype=int)
        sub = c[s:e, :][:, di_arr]
        assert sub.min() >= center_range[0] - 1e-9 and sub.max() <= center_range[1] + 1e-9, \
            "center out of requested range"
        b = beta[s:e]
        assert b.min() >= beta_range[0] - 1e-9 and b.max() <= beta_range[1] + 1e-9, \
            "beta out of requested range"
        k_best = s + int(np.argmin(b))
        x_opt[di_arr] = c[k_best, di_arr]
        f_star += w * float(np.min(b))

    f_val = float(np.squeeze(fitness(x_opt, inst)))
    tol = 1e-4 + 1e-6 * abs(f_star)
    assert abs(f_val - f_star) <= tol, \
        f"internal inconsistency: fitness(x_opt)={f_val} vs f_star={f_star}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a large augmented dataset (Type 3) from a bank of PORTAL recipes."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Recipe JSON files and/or folders (searched recursively).")
    parser.add_argument("--out-dir", required=True, help="Output directory for augmented JSONs.")
    parser.add_argument("--count", type=int, required=True,
                        help="Number of augmented instances per recipe.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed; Type 3 also keys each stream by source-recipe fingerprint.")
    parser.add_argument("--start-sample", type=int, default=1,
                        help="First sample_id (default 1).")
    parser.add_argument("--center-min", type=float, default=CENTER_MIN)
    parser.add_argument("--center-max", type=float, default=CENTER_MAX)
    parser.add_argument("--beta-min", type=float, default=BETA_MIN)
    parser.add_argument("--beta-max", type=float, default=BETA_MAX)
    parser.add_argument("--beta-gap", type=float, default=DEFAULT_BETA_GAP,
                        help="Within-block basin-depth span; one shared base per block "
                             "plus offsets in [0, gap] (default: frozen-catalog value 20).")
    parser.add_argument("--beta-opt-gap", type=float, default=DEFAULT_BETA_OPT_GAP,
                        help="Minimum margin between the block-optimal beta and every "
                             "other beta (default: frozen-catalog value 1).")
    parser.add_argument("--verify", action="store_true",
                        help="Verify each augmented instance (ranges + optimum consistency).")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive.")
    center_range = (float(args.center_min), float(args.center_max))
    beta_range = (float(args.beta_min), float(args.beta_max))
    if center_range[1] < center_range[0] or beta_range[1] < beta_range[0]:
        raise ValueError("Ranges must satisfy min <= max.")

    files = collect_input_files(args.inputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "augmentation_manifest.jsonl"

    print(f"Type 3 augmentation: {len(files)} recipe(s) x {args.count} instance(s)")
    print(f"  center_range={center_range}  beta_range={beta_range}  base_seed={args.base_seed}")

    n_written = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for src in files:
            with contextlib.redirect_stdout(io.StringIO()):
                anchor = load_instance(str(src))
            stem = src.stem
            recipe_seed_key, fingerprint, source_token = recipe_fingerprint(src)
            for j in range(args.start_sample, args.start_sample + args.count):
                inst = realize(
                    anchor,
                    args.base_seed,
                    j,
                    method=METHOD_AUGMENT,
                    center_range=center_range,
                    beta_range=beta_range,
                    beta_gap=float(args.beta_gap),
                    beta_opt_gap=float(args.beta_opt_gap),
                    recipe_seed_key=recipe_seed_key,
                )
                if args.verify:
                    verify_instance(inst, center_range, beta_range)
                # Content and source tokens prevent collisions for equal
                # filenames, including byte-identical inputs from distinct folders.
                out_name = f"{stem}_{fingerprint}_{source_token}_aug_s{j:04d}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    out_path = save_instance(inst, out_name, output_dir=out_dir)
                manifest.write(json.dumps({
                    "source_recipe": str(src),
                    "sample_id": j,
                    "base_seed": args.base_seed,
                    "recipe_seed_key": recipe_seed_key,
                    "recipe_fingerprint": fingerprint,
                    "source_path_token": source_token,
                    "method": METHOD_AUGMENT,
                    "center_range": list(center_range),
                    "beta_range": list(beta_range),
                    "beta_gap": float(args.beta_gap),
                    "beta_opt_gap": float(args.beta_opt_gap),
                    "output": str(out_path),
                }) + "\n")
                n_written += 1
            print(f"  {stem}: {args.count} instances")

    print(f"Done: {n_written} augmented instances -> {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
