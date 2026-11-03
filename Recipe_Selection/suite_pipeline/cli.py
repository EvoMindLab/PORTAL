"""Command-line interface for the PORTAL suite runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .orchestrator import BASE_STAGES, OPTIONAL_TRAIN_TEST_STAGES, SuitePipelineRunner


def _stage_name(value: str) -> str:
    """Accept the historical stage token only as a compatibility alias."""
    return "performance" if value == "isa" else value


def build_parser() -> argparse.ArgumentParser:
    all_stages = BASE_STAGES + OPTIONAL_TRAIN_TEST_STAGES
    parser = argparse.ArgumentParser(
        description=(
            "Construct a PORTAL benchmark suite end to end from a directory of "
            "canonical unified recipe JSONs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="TOML pipeline configuration."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without writing files.",
    )
    parser.add_argument(
        "--from-stage", type=_stage_name, choices=all_stages, default=None, help="Start at this stage."
    )
    parser.add_argument(
        "--to-stage", type=_stage_name, choices=all_stages, default=None, help="Stop after this stage."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun every selected stage."
    )
    parser.add_argument(
        "--force-stage",
        type=_stage_name,
        action="append",
        default=[],
        choices=all_stages,
        help="Rerun one named stage even when compatible outputs exist; repeat as needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config, project_root)
    runner = SuitePipelineRunner(
        config,
        project_root=project_root,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        force=args.force,
        force_stages=args.force_stage,
    )
    runner.run()
    return 0
