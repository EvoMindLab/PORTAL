# Optimizer attribution and licensing

These modules implement published optimization algorithms; the methods are
credited to their original researchers in each module header. They are not
new optimization methods proposed by PORTAL.

## Scope

- `base_algorithm.py` is PORTAL's interface code and follows the root MIT license.
- Other implementation modules are excluded from PORTAL's root MIT grant.
  The existing GPL v3 grant for PORTAL-contributed implementation/adaptation
  code is retained in `LICENSE` here. Original third-party code remains
  subject to its own applicable rights and licenses.
- The CMA-ES and IPOP-CMA-ES wrappers load separately installed pycma/cmaes.
  NiaPy and pymoo are also installed dependencies, accessed through PORTAL
  adapters rather than bundled source trees.

The machine-readable `license_inventory.json` lists every implementation,
its stated provenance, the checked upstream evidence, and release-clearance
status. Scientific references remain in the source headers. A comparison
against upstream code does not by itself establish that code was copied;
the inventory distinguishes paper implementations, package wrappers, and
source-derived implementations under their preserved terms.

## Provenance and public-release scope

The [provenance notes](PROVENANCE.md) distinguish implementation of a published
method from reuse of its source-code expression. On November 3, 2026,
maintainer Danial Yazdani confirmed that the ten previously held implementations
were written from the papers, with reference code consulted for clarification.
Their records identify this as maintainer confirmation, not independent proof
of authorship. The original scientific references and reference-source URLs
are retained; inaccurate direct-translation claims have been removed.

SHADE and L-SHADE have documented author-repository GPL terms, with the exact
license and source credits saved in `../LICENSES/`. No unrelated fork's license
is substituted for original-author permission.

The author-derived `dd-CMA-ES.py` is excluded from the public release because
no explicit source license was found. It was not part of the reported
experiments and was already excluded from Studio's algorithm registry.
All included optimizer numerical routines are unchanged.
