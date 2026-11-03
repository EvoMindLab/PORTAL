"""PyMOO single-objective optimizers exposed through PORTAL's algorithm registry.

PORTAL integration for the separately installed pymoo package, not bundled
pymoo optimizer source. Upstream: https://github.com/anyoptimization/pymoo
Reference: J. Blank and K. Deb (2020), pymoo: Multi-Objective Optimization in
Python, IEEE Access 8, 89497-89509.
https://doi.org/10.1109/ACCESS.2020.2990567

The registry supports single-objective algorithms. Multi-objective methods
require Pareto-front results and corresponding metrics and plots, which
PORTAL's scalar-objective pipeline does not provide.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from base_algorithm import BaseAlgorithm
from portal_studio.algorithms import package_adapters


def get_pymoo_version() -> str | None:
    """Installed pymoo version, or None if the package isn't installed."""
    try:
        return version("pymoo")
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class PymooAlgorithmSpec:
    class_path: str
    category: str
    full_name: str
    pop_size: int = 50
    package: str = "pymoo"


_CATALOG_MODULES: dict[str, tuple[str, ...]] = {
    "Swarm Intelligence": ("PSO",),
    "Evolutionary": ("GA", "DE"),
    "Evolution Strategies": ("ES", "CMA-ES"),
}

_CLASS_PATH_OVERRIDES = {
    "CMA-ES": "pymoo.algorithms.soo.nonconvex.cmaes.CMAES",
}

# pymoo's own class names are bare acronyms (PSO, GA, DE, ES, CMAES) with no
# descriptive text to derive a full name from, unlike NiaPy's CamelCase class
# names -- so these are spelled out explicitly rather than auto-derived.
_FULL_NAMES = {
    "PSO": "Particle Swarm Optimization",
    "GA": "Genetic Algorithm",
    "DE": "Differential Evolution",
    "ES": "Evolution Strategy",
    "CMA-ES": "Covariance Matrix Adaptation Evolution Strategy",
}


def _build_catalog() -> dict[str, PymooAlgorithmSpec]:
    catalog = {}
    for category, names in _CATALOG_MODULES.items():
        for name in names:
            module_stem = name.lower().replace("-", "")
            class_path = _CLASS_PATH_OVERRIDES.get(
                name,
                f"pymoo.algorithms.soo.nonconvex.{module_stem}.{name}",
            )
            catalog[name] = PymooAlgorithmSpec(
                class_path=class_path,
                category=category,
                full_name=_FULL_NAMES[name],
            )
    return catalog


PYMOO_ALGORITHMS = _build_catalog()


def load_pymoo_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001 - normalize optional dependency errors
        raise RuntimeError(
            "pymoo is required for this package-backed algorithm. "
            "Install pymoo or choose another algorithm."
        ) from exc


def make_pymoo_solver(name: str, spec: PymooAlgorithmSpec) -> type:
    class PymooPackageAlgorithm(BaseAlgorithm):
        def __init__(self, pop_size: int = spec.pop_size) -> None:
            super().__init__(name=name)
            self.pop_size = int(pop_size)

        def run(self, problem) -> dict:
            max_fe = int(problem.max_fe)
            if max_fe < 10:
                raise RuntimeError(
                    "pymoo-backed algorithms require at least 10 function "
                    "evaluations."
                )
            algorithm_cls = load_pymoo_class(spec.class_path)
            pop_size = min(max(5, self.pop_size), max_fe)
            package_adapters.run_pymoo_optimizer(
                problem,
                algorithm_cls,
                {"pop_size": pop_size},
                seed=getattr(problem, "algorithm_seed", None),
            )
            return self._make_result(problem)

    PymooPackageAlgorithm.__name__ = (
        f"Pymoo{name.replace('-', '').replace(' ', '')}"
    )
    PymooPackageAlgorithm.__qualname__ = PymooPackageAlgorithm.__name__
    return PymooPackageAlgorithm
