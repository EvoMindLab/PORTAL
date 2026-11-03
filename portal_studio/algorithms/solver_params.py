"""Editable parameter specs for registered solvers.

Hybrid approach: introspect each solver's ``__init__`` signature for names,
defaults, and (best-effort) kinds, then merge a thin per-solver override table
(:data:`PARAM_OVERRIDES`) that supplies ranges, steps, and tooltips. New solvers
render automatically from introspection alone.
"""
from __future__ import annotations

import inspect
import numbers
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from portal_studio.algorithms.registry import REGISTERED_PACKAGE_SPECS, SOLVERS


@dataclass
class ParamSpec:
    name: str
    kind: str                         # 'int' | 'float' | 'bool' | 'choice'
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    decimals: Optional[int] = None
    tooltip: str = ""
    choices: Optional[List[Any]] = None


def _ann_name(annotation: Any) -> Optional[str]:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None)


def _infer_kind(annotation: Any, default: Any) -> str:
    name = _ann_name(annotation)
    # bool first: bool is a subclass of int.
    if name == "bool" or isinstance(default, bool):
        return "bool"
    if name == "str" or isinstance(default, str):
        return "choice"
    if name == "int":
        return "int"
    if name == "float":
        return "float"
    if isinstance(default, numbers.Integral):
        return "int"
    if isinstance(default, numbers.Real):
        return "float"
    return "float"


def _plain_default(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _introspect(cls: type) -> "OrderedDict[str, ParamSpec]":
    specs: "OrderedDict[str, ParamSpec]" = OrderedDict()
    sig = inspect.signature(cls.__init__)
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = None if p.default is inspect.Parameter.empty else _plain_default(p.default)
        specs[name] = ParamSpec(
            name=name,
            kind=_infer_kind(p.annotation, default),
            default=default,
            choices=[default] if isinstance(default, str) else None,
        )
    return specs


PARAM_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "PSO": {
        "pop_size": {"default": 20, "min": 4, "max": 500, "step": 1,
                     "tooltip": "Swarm size."},
        "w_max": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
                  "tooltip": "Initial (maximum) inertia weight."},
        "w_min": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
                  "tooltip": "Final (minimum) inertia weight."},
        "c1": {"min": 0.0, "max": 4.0, "step": 0.1, "decimals": 2,
               "tooltip": "Cognitive (personal-best) acceleration coefficient."},
        "c2": {"min": 0.0, "max": 4.0, "step": 0.1, "decimals": 2,
               "tooltip": "Social (global-best) acceleration coefficient."},
        "v_max_r": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
                    "tooltip": "Velocity clamp as a fraction of the search range."},
    },
    "DE": {
        "pop_size": {"default": 20, "min": 4, "max": 500, "step": 1,
                     "tooltip": "Population size."},
        "F": {"min": 0.0, "max": 2.0, "step": 0.1, "decimals": 2,
              "tooltip": "Differential weight (mutation factor)."},
        "CR": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
               "tooltip": "Crossover probability."},
    },
    "GA": {
        "pop_size": {"min": 4, "max": 1000, "step": 1,
                     "tooltip": "Population size."},
        "eta_c": {"min": 0.0, "max": 100.0, "step": 1.0, "decimals": 1,
                  "tooltip": "SBX crossover distribution index."},
        "eta_m": {"min": 0.0, "max": 100.0, "step": 1.0, "decimals": 1,
                  "tooltip": "Polynomial mutation distribution index."},
        "p_c": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
                "tooltip": "Crossover probability."},
        "p_m": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2,
                "tooltip": "Mutation probability per gene."},
        "elitism": {"tooltip": "Carry the best individual to the next generation."},
    },
    "CMA-ES": {
        "sigma0": {"default": 50.0, "kind": "float", "min": 0.0, "max": 100.0,
                   "step": 0.1, "decimals": 2, "tooltip": "Initial step size."},
        "popsize": {"default": 6, "kind": "int", "min": 4, "max": 500, "step": 1,
                    "tooltip": "Population size."},
        "restarts": {"kind": "int", "min": 0, "max": 20, "step": 1,
                     "tooltip": "Number of IPOP restarts."},
    },
    "Powell": {
        "initial_step_fraction": {"min": 0.0, "max": 1.0, "step": 0.01, "decimals": 3,
                                  "tooltip": "Initial step as a fraction of the search range."},
        "shrink": {"min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
                   "tooltip": "Step-shrink factor after a failed line search."},
        "line_evals": {"min": 1, "max": 100, "step": 1,
                       "tooltip": "Evaluations per line search."},
        # min_step_fraction and tol stay unbounded -> rendered as free-text
        # numeric fields (they are sub-precision tolerances).
        "min_step_fraction": {"tooltip": "Convergence step-size floor (fraction of range)."},
        "tol": {"tooltip": "Objective tolerance for convergence."},
    },
    "FIPS": {
        "topology": {
            "kind": "choice",
            "choices": ["global", "ring"],
            "tooltip": "Neighbourhood topology used by the fully informed particle swarm.",
        },
    },
    "MA-ES": {
        "boundary_handling": {
            "kind": "choice",
            "choices": ["reflect", "clip", "none"],
            "tooltip": "Boundary repair policy applied before evaluating offspring.",
        },
    },
    "SPSO": {
        "pop_size": {"min": 1, "max": 500, "step": 1,
                     "tooltip": "Swarm size."},
        "informants": {"min": 1, "max": 100, "step": 1,
                       "tooltip": "Number of random informants per particle."},
        "w": {"min": 0.0, "max": 1.5, "step": 0.01, "decimals": 10,
              "tooltip": "Inertia coefficient."},
        "c": {"min": 0.0, "max": 4.0, "step": 0.01, "decimals": 10,
              "tooltip": "Acceleration coefficient for SPSO-2011 sampling."},
    },
}


def get_solver_params(solver_name: str) -> "OrderedDict[str, ParamSpec]":
    """Ordered editable parameter specs for ``solver_name`` (empty if none)."""
    cls = SOLVERS[solver_name]
    specs = _introspect(cls)
    overrides = PARAM_OVERRIDES.get(solver_name, {})
    for pname, fields in overrides.items():
        if pname not in specs:
            continue
        spec = specs[pname]
        for key, value in fields.items():
            setattr(spec, key, value)
    if solver_name in REGISTERED_PACKAGE_SPECS and "pop_size" in specs:
        pop_size = specs["pop_size"]
        pop_size.min = 5
        pop_size.max = 10000
        pop_size.step = 1
        if not pop_size.tooltip:
            pop_size.tooltip = "Population size required by the package-backed optimizer."
    return specs
