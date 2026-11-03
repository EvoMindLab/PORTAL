"""Registry of solvers exposed to PORTAL Studio.

Solvers are auto-discovered from the repo-root ``algorithms/`` folder: every
file there that defines a :class:`BaseAlgorithm` subclass is registered, so
adding a new solver is just dropping a conforming file in that folder. The
dropdown lists every solver alphabetically. Files that fail to import are
skipped and recorded in :data:`DISCOVERY_SKIPPED`.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
import os
import sys
from typing import Dict, List, Tuple, Type

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ALG_DIR = os.path.join(_REPO, "algorithms")
for p in (_REPO, _ALG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from base_algorithm import BaseAlgorithm                          # noqa: E402
from portal_studio.algorithms.niapy_catalog import (              # noqa: E402
    NIAPY_ALGORITHMS,
    get_niapy_version,
    make_niapy_solver,
)
from portal_studio.algorithms.pymoo_catalog import (              # noqa: E402
    PYMOO_ALGORITHMS,
    get_pymoo_version,
    make_pymoo_solver,
)
from portal_studio.algorithms.random_search import RandomSearch   # noqa: E402

# Files in algorithms/ that must not be auto-registered as solvers.
_SKIP_FILES = {
    "base_algorithm.py",      # the abstract base, not a solver
    "portal_problem.py",      # problem adapter, not a solver
    "__init__.py",
    "CMA-ES.py",              # duplicate of CMA_ES.py (same CMA_ES class)
    "dd-CMA-ES.py",          # raw implementation, not a BaseAlgorithm wrapper
}

# Display-label overrides for solver filenames.
_DISPLAY_OVERRIDES = {
    "CMA_ES.py": "CMA-ES",
    "SPSO-2011.py": "SPSO",
}

# Populated by _discover(): list of "filename (reason)" for files that were
# present but could not be registered (missing deps, no wrapper class, etc.).
DISCOVERY_SKIPPED: List[str] = []
REGISTERED_PACKAGE_SPECS = {}


def _load_wrapper(path: str) -> Tuple[type, ...]:
    """Import one algorithm file and return its BaseAlgorithm subclass, if any.

    Uses an explicit file-path import so hyphenated filenames (e.g.
    ``LSHADE-RSP.py``) load too. Raises on import failure; the caller decides
    whether to skip.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    modname = "portal_alg_" + stem.replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return tuple(
        obj for _name, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, BaseAlgorithm) and obj is not BaseAlgorithm
        and obj.__module__ == modname
    )


def _discover() -> "Dict[str, Type]":
    found: Dict[str, Type] = {"Random Search": RandomSearch}
    DISCOVERY_SKIPPED.clear()
    REGISTERED_PACKAGE_SPECS.clear()
    for fname in sorted(os.listdir(_ALG_DIR)):
        if not fname.endswith(".py") or fname in _SKIP_FILES:
            continue
        display = _DISPLAY_OVERRIDES.get(fname, os.path.splitext(fname)[0])
        try:
            wrappers = _load_wrapper(os.path.join(_ALG_DIR, fname))
        except Exception as exc:  # missing optional dep, syntax error, etc.
            DISCOVERY_SKIPPED.append(f"{fname} ({type(exc).__name__}: {exc})")
            continue
        if not wrappers:
            DISCOVERY_SKIPPED.append(f"{fname} (no BaseAlgorithm subclass)")
            continue
        found[display] = wrappers[0]
    for display, spec in NIAPY_ALGORITHMS.items():
        if display in found:
            display = f"NIAPY {display}"
        found[display] = make_niapy_solver(display, spec)
        REGISTERED_PACKAGE_SPECS[display] = spec
    for display, spec in PYMOO_ALGORITHMS.items():
        if display in found:
            display = f"PYMOO {display}"
        found[display] = make_pymoo_solver(display, spec)
        REGISTERED_PACKAGE_SPECS[display] = spec
    # All solvers listed alphabetically (case-insensitive).
    return {n: found[n] for n in sorted(found, key=str.lower)}


SOLVERS: Dict[str, Type] = _discover()


@dataclass(frozen=True)
class AlgorithmMeta:
    name: str
    category: str
    source: str = "PORTAL"
    implementation: str = ""
    description: str = ""


_CATEGORY_OVERRIDES = {
    "APSO": "Swarm Intelligence",
    "BBPSO": "Swarm Intelligence",
    "CLPSO": "Swarm Intelligence",
    "FIPS": "Swarm Intelligence",
    "GLPSO": "Swarm Intelligence",
    "HCLPSO": "Swarm Intelligence",
    "PSO": "Swarm Intelligence",
    "QPSO": "Swarm Intelligence",
    "SLPSO": "Swarm Intelligence",
    "SPSO": "Swarm Intelligence",
    "TAPSO": "Swarm Intelligence",
    "DE": "Evolutionary",
    "GA": "Evolutionary",
    "JADE": "Evolutionary",
    "jDE": "Evolutionary",
    "jSO": "Evolutionary",
    "L-SRTDE": "Evolutionary",
    "LSHADE": "Evolutionary",
    "LSHADE-cnEpSin": "Evolutionary",
    "LSHADE-RSP": "Evolutionary",
    "NL-SHADE-RSP": "Evolutionary",
    "SaDE": "Evolutionary",
    "SHADE": "Evolutionary",
    "CMA-ES": "Evolution Strategies",
    "CMSA-ES": "Evolution Strategies",
    "EA4eig": "Evolution Strategies",
    "ES": "Evolution Strategies",
    "MA-ES": "Evolution Strategies",
    "NES": "Evolution Strategies",
    "MadDE": "Evolutionary",
    "IMODE": "Evolutionary",
    "PatternSearch": "Classical",
    "Powell": "Classical",
    "Random Search": "Classical",
}

_SOURCE_OVERRIDES: dict[str, str] = {}

_CATEGORY_ORDER = [
    "Swarm Intelligence",
    "Evolutionary",
    "Evolution Strategies",
    "Classical",
    "Basic",
    "Modified",
    "Other",
]

# Display labels for AlgorithmMeta.source, in the order they should appear in
# a source selector. Internal source values are the backing package's own
# name ("niapy", "pymoo") — this mapping only affects GUI display.
_SOURCE_DISPLAY = {
    "PORTAL": "PORTAL",
    "niapy": "NIAPY",
    "pymoo": "PYMOO",
}
_SOURCE_ORDER = ["PORTAL", "NIAPY", "PYMOO"]


def _native_full_name(cls: type) -> str:
    """Best-effort full/expanded name for a native wrapper, taken from its
    module docstring's first line (native files document themselves as
    "Short Name -- Description" or similar). Falls back to the class's own
    ``name`` attribute set in ``BaseAlgorithm.__init__`` when no usable
    docstring is found, so callers always get a sensible display string."""
    module = inspect.getmodule(cls)
    doc = (getattr(module, "__doc__", None) or "").strip()
    first_line = doc.split("\n", 1)[0].strip() if doc else ""
    for sep in (" — ", " -- ", " - "):
        if sep in first_line:
            return first_line.split(sep, 1)[1].strip()
    return first_line


def _build_metadata() -> dict[str, AlgorithmMeta]:
    metadata = {}
    for name in SOLVERS:
        package_spec = REGISTERED_PACKAGE_SPECS.get(name)
        category = _CATEGORY_OVERRIDES.get(
            name,
            package_spec.category if package_spec is not None else "Other",
        )
        if package_spec is not None:
            full_name = package_spec.full_name
        else:
            full_name = _native_full_name(SOLVERS[name]) or name
        metadata[name] = AlgorithmMeta(
            name=name,
            category=category,
            source=_SOURCE_OVERRIDES.get(
                name,
                package_spec.package if package_spec is not None else "PORTAL",
            ),
            implementation=getattr(SOLVERS[name], "__module__", ""),
            description=full_name,
        )
    return metadata


ALGORITHM_METADATA = _build_metadata()

_SOURCE_VERSION_LOOKUP = {
    "NIAPY": get_niapy_version,
    "PYMOO": get_pymoo_version,
}


def get_source_version(source: str) -> str | None:
    """Installed version of the package backing ``source`` (a display label
    from :func:`list_algorithm_sources`), or None for "PORTAL" (native, no
    package) or if the package isn't installed."""
    lookup = _SOURCE_VERSION_LOOKUP.get(source)
    return lookup() if lookup is not None else None


def list_algorithm_sources() -> list[str]:
    """Return the display-level source labels ("PORTAL", "NIAPY", ...)."""
    present = {
        _SOURCE_DISPLAY.get(meta.source, meta.source)
        for meta in ALGORITHM_METADATA.values()
    }
    ordered = [source for source in _SOURCE_ORDER if source in present]
    extra = sorted(present.difference(ordered), key=str.lower)
    return ordered + extra


def list_algorithm_categories(source: str | None = None) -> list[str]:
    present = {
        meta.category for meta in ALGORITHM_METADATA.values()
        if source is None or _SOURCE_DISPLAY.get(meta.source, meta.source) == source
    }
    ordered = [category for category in _CATEGORY_ORDER if category in present]
    extra = sorted(present.difference(ordered), key=str.lower)
    return ordered + extra


def list_solvers(category: str | None = None,
                  source: str | None = None) -> List[str]:
    """Return the user-facing names in canonical Education-tab order."""
    if category is None and source is None:
        return list(SOLVERS.keys())
    names = []
    for name in SOLVERS:
        meta = ALGORITHM_METADATA.get(name, AlgorithmMeta(name, "Other"))
        if category is not None and meta.category != category:
            continue
        if source is not None and _SOURCE_DISPLAY.get(meta.source, meta.source) != source:
            continue
        names.append(name)
    return names


def get_algorithm_meta(name: str) -> AlgorithmMeta:
    return ALGORITHM_METADATA[name]
