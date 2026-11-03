"""Logging configuration utilities."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional


def configure_logging(path: Optional[str] = None) -> None:
    """Configure logging from YAML file if available; fall back to basicConfig.

    Parameters
    ----------
    path : str or None
        Path to YAML logging config. If missing or YAML unavailable, uses defaults.
    """

    if path:
        p = Path(path)
        if p.exists():
            try:
                import yaml

                with p.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                logging.config.dictConfig(cfg)
                return
            except Exception:
                # Fall through to basicConfig on any error
                pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
