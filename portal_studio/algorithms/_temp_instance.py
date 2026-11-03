"""Context manager that materialises an in-memory PORTAL instance dict to a
tempfile that ``PortalProblem`` can load by path, then deletes it on exit.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator

from portal_core.save_instance import save_instance, save_composite_as_unified


@contextmanager
def instance_as_tempfile(instance_dict: Dict[str, Any]) -> Generator[str, None, None]:
    is_composite = bool(
        instance_dict.get("composite", False)
        or instance_dict.get("_composite", False)
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    path = tmp.name
    try:
        if is_composite:
            # A multi-block (Layer-3) staging structure is flattened to a single
            # unified instance, never a composite envelope: the run pipeline
            # always loads one unified JSON (PortalProblem -> load_instance),
            # matching Studio's saved instances.
            save_composite_as_unified(instance_dict, filename=os.path.basename(path),
                                      output_dir=os.path.dirname(path) or ".")
        else:
            save_instance(instance_dict, filename=os.path.basename(path),
                          output_dir=os.path.dirname(path) or ".")
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
