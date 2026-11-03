"""Rebuild the small, explicitly categorized 2D curation demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from portal_core.benchmark_generator import benchmark_generator, precompute_scalings
from portal_core.save_instance import save_instance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "recipes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plans = [
        ("Baseline", 0.4, 1.0, None),
        ("Baseline", 0.6, 5.0, None),
        ("Baseline", 1.0, 20.0, None),
        ("Baseline", 1.4, 50.0, None),
        ("Transformed", 0.4, 2.0, "additive_periodic"),
        ("Transformed", 0.7, 10.0, "log_sinusoidal"),
        ("Transformed", 1.0, 25.0, "wavelet_mod"),
        ("Transformed", 1.3, 60.0, "tensor_interference"),
    ]
    for index, (family, exponent, conditioning, transform) in enumerate(plans, 1):
        recipe_id = f"demo_{index:02d}"
        target = args.output / f"{recipe_id}.json"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; choose another output or --overwrite.")
        settings = dict(
            Seed=500 + index, Dimension=2, NumComponents=1,
            BaselineMode=3, BaselineType="A", PMode=2, FixedP=exponent,
            KappaMode=4,
            RotationMode=1 if index % 2 else 2,
            TransformMode=3 if transform is None else 4,
        )
        if transform:
            settings["TransformSeq"] = [[transform]]
        instance = benchmark_generator(**settings)
        # Explicitly set the two directional factors, then recalibrate the
        # derived coefficients; the generator has no kappa_min/max kwargs.
        instance["P"]["kappa_plus"][0, :] = [1.0, conditioning]
        instance["P"]["kappa_minus"][0, :] = [1.0, conditioning]
        form, single, plus, minus = precompute_scalings(instance)
        instance["P"].update(form=form, rho_single=single, rho_plus=plus, rho_minus=minus)
        instance["_suite"] = {"family": family, "recipe_id": recipe_id}
        save_instance(instance, filename=target.name, output_dir=str(args.output))
    print(f"Prepared {len(plans)} demonstration recipes in {args.output}")


if __name__ == "__main__":
    main()
