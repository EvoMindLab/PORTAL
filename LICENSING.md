# Licensing scope

## PORTAL-owned material: MIT

The root LICENSE applies to PORTAL-owned framework code and original
PORTAL documentation, examples, recipe definitions, and tests. This includes
`portal_core/`, `portal_studio/` (including PORTAL's optimizer adapters),
`Recipe_Selection/`, the root Python entry points, and
`algorithms/base_algorithm.py`, subject to any specific third-party notices.

Copyright (c) 2025-2026 Danial Yazdani and PORTAL contributors.
Existing contributor credits remain in place. This notice does not claim
ownership of third-party implementations or invention of the cited algorithms.

Research manuscripts, published papers, study-specific datasets, and other
research artifacts are outside this software license grant unless expressly
licensed otherwise. They are not part of the public software export.

## Optimizer implementations: separate terms

The optimizer implementation modules in `algorithms/`, other than the
PORTAL interface `base_algorithm.py`, are outside the root MIT grant.
The pre-existing GPL v3 grant for PORTAL-contributed implementation and
adaptation code is retained in `algorithms/LICENSE`. It does not supply
missing permission for original third-party code or change an upstream
license. Original authors, scientific references, and source credits remain
in the module headers.

See `algorithms/README.md` and `algorithms/license_inventory.json` for
per-file provenance, upstream licensing evidence, and public-release exclusions.
A published method and a source-code implementation are different: a paper
citation is not a source-code redistribution license.

## Bundled and installed third-party software

The adapted `pflacco/` subset retains Raphael Patrick Prager's MIT notice
in `pflacco/LICENSE`; it is not relicensed in Danial Yazdani's name.
Separately installed packages, including pycma, cmaes, NiaPy, pymoo, Qt/PySide6,
and PuLP/CBC, retain their own licenses. Their inclusion or use is not covered
by a new MIT grant from PORTAL. Preserve the required upstream notices and
comply with any applicable copyleft or redistribution obligations.

Separate notices do not eliminate license obligations for a combined work.
In particular, do not describe the complete distribution with GPL-covered
optimizers as entirely MIT-licensed. The MIT grant applies to the PORTAL-owned
material identified above.

## Release status

The follow-up review documented GPL terms for SHADE and L-SHADE and preserved
the author repository's license and source credits in `LICENSES/`.
Maintainer Danial Yazdani confirmed on November 3, 2026 that the ten previously
held implementations were developed from the papers, with author code used
for clarification. Their provenance is recorded as maintainer-confirmed rather
than independently verified authorship. Scientific and source credits remain.

The author-derived `algorithms/dd-CMA-ES.py` has no located license grant and
is explicitly excluded from the public release. It remains in the private
research checkout only; the export does not include or relicense it.
See `algorithms/PROVENANCE.md` and the per-file inventory.

Earlier GPL releases remain available under their original terms. This change
does not revoke previously granted rights or rewrite repository history.
