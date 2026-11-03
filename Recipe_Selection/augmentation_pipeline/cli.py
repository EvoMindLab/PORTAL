"""Command-line interface for Type 3 dataset augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import AugmentationDatasetRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mass-produce canonical PORTAL JSON instances from a source recipe "
            "bank using Type 3 augmentation. This command does not run ELA or Performance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="TOML augmentation configuration."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved plan without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace a prior managed run in the configured output directory. "
            "Unmarked non-empty directories are never deleted."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config, project_root)
    runner = AugmentationDatasetRunner(
        config,
        project_root=project_root,
        dry_run=args.dry_run,
        force=args.force,
    )
    runner.run()
    return 0
