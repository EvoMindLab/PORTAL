"""PORTAL command-line interface.

Subcommands:

  Evaluation (over portal_core):
    portal compute            --input X.csv --instance  inst.json --output y.csv
    portal compute-composite  --input X.csv --composite instance.json --output y.csv

  Curation (wraps the run_*.py scripts in Recipe_Selection/):
    portal pipeline           # run ELA then Performance (mirrors run_pipeline.py)
    portal ela                # run ELA only       (wraps run_ela.py)
    portal performance                # run Performance only       (wraps run_curation_solvers.py)

Each curation subcommand passes any extra arguments through to the
underlying script unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from ._io import read_array, write_array
from ._logging import configure_logging
from . import __version__
from portal_core.fitness import fitness
from portal_core.load_instance import load_instance


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CURATION_ROOT = _REPO_ROOT / "Recipe_Selection"


# -------------------- evaluation subcommands ---------------------------------

def _evaluate_rows(X: np.ndarray, fn) -> np.ndarray:
    return np.array([float(fn(X[i])) for i in range(X.shape[0])], dtype=np.float64)


def _cmd_compute(args: argparse.Namespace) -> int:
    configure_logging("config/logging.yaml")
    X = read_array(args.input, args.input_format)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    inst = load_instance(args.instance)
    y = _evaluate_rows(X, lambda row: fitness(row, inst))
    out = y if not args.as_column else y.reshape(-1, 1)
    write_array(args.output, out, args.output_format)
    return 0


def _cmd_compute_composite(args: argparse.Namespace) -> int:
    """Backward-compatible command name for unified multi-block evaluation."""
    configure_logging("config/logging.yaml")
    X = read_array(args.input, args.input_format)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    instance = load_instance(args.composite)
    y = _evaluate_rows(X, lambda row: fitness(row, instance))
    out = y if not args.as_column else y.reshape(-1, 1)
    write_array(args.output, out, args.output_format)
    return 0


def _add_eval_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="Path to input dataset (n_samples x d)")
    p.add_argument("--output", required=True, help="Where to write outputs")
    p.add_argument("--input-format",  dest="input_format",  default=None,
                   help="Input format override: csv|npy|npz|json")
    p.add_argument("--output-format", dest="output_format", default=None,
                   help="Output format override: csv|npy|npz|json")
    p.add_argument("--as-column", action="store_true",
                   help="Save outputs as a single column (2-D)")


# -------------------- curation subcommands -----------------------------------

def _run_script(script_name: str, extra_args: List[str]) -> int:
    """Run one curation script from ``Recipe_Selection`` as a subprocess."""
    script = _CURATION_ROOT / script_name
    if not script.exists():
        sys.stderr.write(f"curation script not found: {script}\n")
        return 1
    cmd = [sys.executable, str(script), *extra_args]
    return subprocess.call(cmd, cwd=str(_CURATION_ROOT))


def _cmd_pipeline(args: argparse.Namespace) -> int:
    return _run_script("run_pipeline.py", args.extra)


def _cmd_ela(args: argparse.Namespace) -> int:
    return _run_script("run_ela.py", args.extra)


def _cmd_performance(args: argparse.Namespace) -> int:
    return _run_script("run_curation_solvers.py", args.extra)


# -------------------- entry point --------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="portal",
                                     description="PORTAL CLI (Studio + Curation)")
    parser.add_argument("--version", action="version", version=f"PORTAL {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    p_c = sub.add_parser("compute",
                         help="Evaluate a unified single- or multi-block PORTAL JSON")
    _add_eval_io_args(p_c)
    p_c.add_argument("--instance", required=True,
                     help="Path to a unified PORTAL instance JSON")
    p_c.set_defaults(func=_cmd_compute)

    p_cc = sub.add_parser("compute-composite",
                          help="Legacy command name for evaluating a unified multi-block JSON")
    _add_eval_io_args(p_cc)
    p_cc.add_argument("--composite", required=True,
                      help="Path to a unified multi-block PORTAL JSON")
    p_cc.set_defaults(func=_cmd_compute_composite)

    p_pl = sub.add_parser("pipeline",
                          help="Run ELA then Performance over the recipe Suite (wraps run_pipeline.py)")
    p_pl.add_argument("extra", nargs=argparse.REMAINDER,
                      help="Arguments passed through to run_pipeline.py")
    p_pl.set_defaults(func=_cmd_pipeline)

    p_el = sub.add_parser("ela",
                          help="Extract raw ELA features per recipe (wraps run_ela.py)")
    p_el.add_argument("extra", nargs=argparse.REMAINDER,
                      help="Arguments passed through to run_ela.py")
    p_el.set_defaults(func=_cmd_ela)

    p_is = sub.add_parser("performance",
                          help="Run the solver portfolio for Performance (wraps run_curation_solvers.py)")
    p_is.add_argument("extra", nargs=argparse.REMAINDER,
                      help="Arguments passed through to run_curation_solvers.py")
    p_is.set_defaults(func=_cmd_performance)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "isa":  # Historical CLI token, not a method name.
        argv[0] = "performance"
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
