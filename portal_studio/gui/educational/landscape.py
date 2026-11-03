"""2D grid and landscape-computation helpers for the Education tab."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from portal_core.compose_blocks import fitness_composite_batch
from portal_core.fitness import fitness_batch
from portal_core.instance_factory import component_config_to_instance, merge_k1_instances
from portal_studio.session import ComponentSpec


@lru_cache(maxsize=8)
def grid_2d(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-100.0, 100.0, int(n))
    ys = np.linspace(-100.0, 100.0, int(n))
    Xg, Yg = np.meshgrid(xs, ys)
    X = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    for arr in (Xg, Yg, X):
        arr.setflags(write=False)
    return Xg, Yg, X


def _component_to_instance(spec: ComponentSpec) -> dict:
    return component_config_to_instance(
        {
            "baseline": spec.baseline,
            "rotation_deg": spec.rotation_deg,
            "neutralize": spec.neutralize,
            "params": spec.params,
            "T_seq": spec.T_seq,
        },
        d=2,
    )


def compute_block_landscape(components: list[ComponentSpec], X: np.ndarray) -> np.ndarray:
    insts = [_component_to_instance(component) for component in components]
    multi = merge_k1_instances(insts)
    return np.asarray(fitness_batch(X, multi, evaluation_device="numpy"), dtype=float)


def compute_instance_landscape(instance: dict, X: np.ndarray) -> np.ndarray:
    if instance.get("composite"):
        return np.asarray(
            fitness_composite_batch(X, instance, evaluation_device="numpy"),
            dtype=float,
        )
    return np.asarray(fitness_batch(X, instance, evaluation_device="numpy"), dtype=float)


def compute_component_landscape(spec: ComponentSpec, X: np.ndarray) -> np.ndarray:
    inst = _component_to_instance(spec)
    return np.asarray(fitness_batch(X, inst, evaluation_device="numpy"), dtype=float)
