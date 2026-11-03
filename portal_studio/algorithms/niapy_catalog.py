"""NiaPy optimizers exposed through PORTAL's algorithm registry.

PORTAL integration for the separately installed NiaPy package, not bundled
NiaPy optimizer source. Upstream: https://github.com/NiaOrg/NiaPy
Reference: G. Vrbancic et al. (2018), NiaPy: Python microframework for building
nature-inspired algorithms, JOSS 3(23), 613.
https://doi.org/10.21105/joss.00613

Algorithm classes exported by NiaPy's ``basic``, ``modified``, and ``other``
sub-packages are registered automatically, except those documented in
``_EXCLUDED_CLASS_NAMES``. Selector codes are derived from class-name capitals
and digits by ``_acronym_for_class_name``, with explicit overrides for
evolution strategies. Tooltips show the full algorithm names.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from base_algorithm import BaseAlgorithm
from portal_studio.algorithms import package_adapters


def get_niapy_version() -> str | None:
    """Installed niapy version, or None if the package isn't installed."""
    try:
        return version("niapy")
    except PackageNotFoundError:
        return None


def _humanize_class_name(class_name: str) -> str:
    """"ArtificialBeeColonyAlgorithm" -> "Artificial Bee Colony Algorithm"."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", class_name)
    s = re.sub(r"(?<=[a-z])(?=[0-9])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s


def _acronym_for_class_name(class_name: str) -> str:
    """"DifferentialEvolution" -> "DE"; "MultipleTrajectorySearchV1" ->
    "MTSV1". Mechanical (every capital letter and digit, in order) so it
    needs no per-algorithm table, matching how PORTAL's own solver files
    and PyMOO's catalog are named (short code in the selector, full name in
    the tooltip)."""
    return "".join(ch for ch in class_name if ch.isupper() or ch.isdigit())


# Manual override for the four Evolution Strategy variants, whose names
# encode a standard ES notation (mu, lambda, "+"/"," selection): the
# mechanical acronym rule above collapses two of them to the same string
# ("EvolutionStrategyMpL" and "EvolutionStrategyML" both reduce to "ESML"),
# and explicit labels preserve the standard selection notation.
_SHORT_NAME_OVERRIDES = {
    "EvolutionStrategy1p1": "ES(1+1)",
    "EvolutionStrategyMp1": "ES(mu+1)",
    "EvolutionStrategyMpL": "ES(mu+lambda)",
    "EvolutionStrategyML": "ES(mu,lambda)",
}
_FULL_NAME_OVERRIDES = {
    "EvolutionStrategy1p1": "Evolution Strategy (1 + 1)",
    "EvolutionStrategyMp1": "Evolution Strategy (mu + 1)",
    "EvolutionStrategyMpL": "Evolution Strategy (mu + lambda)",
    "EvolutionStrategyML": "Evolution Strategy (mu, lambda)",
}


@dataclass(frozen=True)
class NiapyAlgorithmSpec:
    class_path: str
    category: str
    full_name: str
    pop_size: int = 50
    package: str = "niapy"


# NiaPy classes intentionally excluded from the PORTAL Studio registry.
_EXCLUDED_CLASS_NAMES = {
    "FireflyAlgorithm", "BatAlgorithm", "HarmonySearch", "HarmonySearchV1",
    "CuckooSearch",
    "GreyWolfOptimizer", "MothFlameOptimizer", "SineCosineAlgorithm",
    "HarrisHawksOptimization",
    "ArtificialBeeColonyAlgorithm", "BeesAlgorithm",
    "BacterialForagingOptimization", "CamelAlgorithm",
    "CoralReefsOptimization", "CatSwarmOptimization",
    "ForestOptimizationAlgorithm", "FlowerPollinationAlgorithm",
    "FishSchoolSearch", "FireworksAlgorithm", "EnhancedFireworksAlgorithm",
    "DynamicFireworksAlgorithm", "DynamicFireworksAlgorithmGauss",
    "BareBonesFireworksAlgorithm", "GlowwormSwarmOptimization",
    "GlowwormSwarmOptimizationV1", "GlowwormSwarmOptimizationV2",
    "GlowwormSwarmOptimizationV3", "KrillHerd",
    "MonarchButterflyOptimization", "GravitationalSearchAlgorithm",
    "AnarchicSocietyOptimization", "ClonalSelectionAlgorithm",
    "LionOptimizationAlgorithm", "MonkeyKingEvolutionV1",
    "MonkeyKingEvolutionV2", "MonkeyKingEvolutionV3",
    "MantisSearchAlgorithm", "MShOA",
    "HybridBatAlgorithm", "HybridSelfAdaptiveBatAlgorithm",
    "ParameterFreeBatAlgorithm", "AdaptiveBatAlgorithm",
    "SelfAdaptiveBatAlgorithm",
}

_GROUP_CATEGORY = {"basic": "Basic", "modified": "Modified", "other": "Other"}


def _discover_algorithm_classes() -> list[tuple[str, type, str]]:
    """Every (class name, class, category) NiaPy exports that isn't
    excluded. Pulled straight from each sub-package's own ``__all__``, so a
    niapy upgrade that adds new algorithms picks them up automatically; a
    non-class export (e.g. a mutation-operator helper function) is skipped."""
    found = []
    for group, category in _GROUP_CATEGORY.items():
        module = import_module(f"niapy.algorithms.{group}")
        for class_name in getattr(module, "__all__", []):
            if class_name in _EXCLUDED_CLASS_NAMES:
                continue
            obj = getattr(module, class_name)
            if not inspect.isclass(obj):
                continue
            found.append((class_name, obj, category))
    return found


def _build_catalog() -> dict[str, NiapyAlgorithmSpec]:
    catalog: dict[str, NiapyAlgorithmSpec] = {}
    try:
        classes = _discover_algorithm_classes()
    except ImportError:
        # NiaPy is optional; native solvers must remain available without it.
        return catalog
    for class_name, cls, category in classes:
        short_name = _SHORT_NAME_OVERRIDES.get(
            class_name, _acronym_for_class_name(class_name)
        )
        full_name = _FULL_NAME_OVERRIDES.get(
            class_name, _humanize_class_name(class_name)
        )
        catalog[short_name] = NiapyAlgorithmSpec(
            class_path=f"{cls.__module__}.{class_name}",
            category=category,
            full_name=full_name,
        )
    return catalog


NIAPY_ALGORITHMS = _build_catalog()


def load_niapy_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001 - normalize optional dependency errors
        raise RuntimeError(
            "niapy is required for this package-backed algorithm. "
            "Install niapy or choose another algorithm."
        ) from exc


def make_niapy_solver(name: str, spec: NiapyAlgorithmSpec) -> type:
    class NiapyPackageAlgorithm(BaseAlgorithm):
        def __init__(self, pop_size: int = spec.pop_size) -> None:
            super().__init__(name=name)
            self.pop_size = int(pop_size)

        def run(self, problem) -> dict:
            max_fe = int(problem.max_fe)
            if max_fe < 10:
                raise RuntimeError(
                    "niapy-backed algorithms require at least 10 function "
                    "evaluations."
                )
            algorithm_cls = load_niapy_class(spec.class_path)
            pop_size = min(max(5, self.pop_size), max_fe)
            package_adapters.run_niapy_optimizer(
                problem,
                algorithm_cls,
                {"population_size": pop_size},
                seed=getattr(problem, "algorithm_seed", None),
            )
            return self._make_result(problem)

    NiapyPackageAlgorithm.__name__ = (
        f"Niapy{re.sub(r'[^0-9A-Za-z]', '', name)}"
    )
    NiapyPackageAlgorithm.__qualname__ = NiapyPackageAlgorithm.__name__
    return NiapyPackageAlgorithm
