# Third-party code and dependencies

## PORTAL's MIT grant is limited to PORTAL-owned material

The root `LICENSE` names Danial Yazdani and PORTAL contributors and licenses
PORTAL-owned framework material under MIT. It does not claim ownership of,
or replace the licenses of, bundled optimizers or external packages.
See [LICENSING.md](LICENSING.md) for the exact scope.

## Bundled pflacco routines

The `pflacco/` directory contains an adapted subset of the native-Python
landscape-feature implementation by Raphael Patrick Prager and contributors:
<https://github.com/Reiyan/pflacco>. Its original MIT copyright and permission
notice is retained in `pflacco/LICENSE`. PORTAL includes local sampling and
numerical-compatibility adaptations; this is not an unmodified upstream copy.

Software reference: Raphael Patrick Prager and Heike Trautmann (2024),
"Pflacco: Feature-Based Landscape Analysis of Continuous and Constrained
Optimization Problems in Python," Evolutionary Computation 32(3), 211-216.
<https://doi.org/10.1162/evco_a_00341>.

The vendored modules and PORTAL's ELA integration layers retain source
credits and software citations. Original per-feature references remain intact.
The upstream license was checked at
<https://github.com/Reiyan/pflacco/blob/master/LICENSE>.

## Optimizer implementations and adapters

The modules in `algorithms/` implement methods developed by the researchers
identified in their headers. Those references credit the methods; they are
not claims that PORTAL invented the optimizers. The method implementations,
except the PORTAL interface `base_algorithm.py`, are outside the root MIT grant.

The previous GPL v3 grant for PORTAL-contributed implementation/adaptation
code is retained in `algorithms/LICENSE`. Original third-party rights and
terms are not replaced by that local grant. Consult the
[optimizer notes](algorithms/README.md) and
[per-file inventory](algorithms/license_inventory.json).

Licensing review on November 3, 2026 found:

- `MadDE.py` declares translation of the authors' MATLAB source, available at
  <https://github.com/subhodipbiswas/MadDE/blob/main/MadDE/MadDE.m>.
  That author repository publishes GPL v3 terms in
  <https://github.com/subhodipbiswas/MadDE/blob/main/LICENSE>.
- The CMA-ES wrappers import pycma rather than bundle its numerical core.
  The package uses BSD 3-Clause terms, while its `purecma.py` contains an
  explicit public-domain declaration. Source:
  <https://github.com/CMA-ES/pycma>.
- IPOP-CMA-ES imports the MIT-licensed cmaes package:
  <https://github.com/CyberAgentAILab/cmaes/blob/main/LICENSE>.
- The L-SRTDE author reference code is GPL v3:
  <https://github.com/VladimirStanovov/L-SRTDE_CEC-2024/blob/main/LICENSE>.
  The local module describes an independent paper implementation checked
  against that source, not a claim to relicense the author's C++ code.
- The APSO comparison implementation is MIT:
  <https://github.com/ZongSingHuang/Adaptive-Particle-Swarm-Optimization/blob/master/LICENSE>.
  The local header states that it was used for cross-checking.
- The exact SHADE 1.1.1 and L-SHADE 1.0.1 author archives were located in
  Ryoji Tanabe's repository, which declares GPL v3. Its exact license and
  the Tanabe/Fukunaga and Jingqiao Zhang utility credits are preserved in
  `LICENSES/Tanabe-GPL-3.0.txt` and `LICENSES/Tanabe-NOTICE.txt`.
- Maintainer Danial Yazdani confirmed that the ten implementations previously
  held for provenance review were written from the papers, with reference
  code consulted for clarification. The inventory records that confirmation
  explicitly; scientific references and reference-source credits remain.

The author-derived `algorithms/dd-CMA-ES.py` is not included in the public
release. Its numerical core originated in Youhei Akimoto's implementation:
<https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518>.
No explicit license was found in the checked gist, so the maintainer approved
excluding that module rather than redistributing it. This exclusion does not
affect Studio's registered optimizers or the reported experiments. No included
optimizer numerical routine was changed by the licensing review.

## Separately installed dependencies

These packages are installed by the user, not copied into PORTAL's source
tree. Their own distributions provide the applicable terms and notices.

| Dependency | Checked license information |
|---|---|
| pycma | BSD 3-Clause package; `purecma.py` has a public-domain declaration |
| cmaes | MIT |
| NiaPy | MIT |
| pymoo | Apache 2.0 |
| PySide6 / Qt | LGPL/GPL or commercial options; comply with the terms of the Qt modules actually used |
| PuLP | MIT; the separately supplied CBC backend has its own terms |
| NumPy, SciPy, pandas, scikit-learn, threadpoolctl, numdifftools | Their published BSD-family/component notices |
| Matplotlib | Its own PSF-style license and included third-party notices |
| SALib, openpyxl, PyYAML | MIT |
| Optional PyTorch | Its own BSD-style and bundled third-party notices |

Primary package sources:
<https://github.com/NiaOrg/NiaPy/blob/master/LICENSE>,
<https://github.com/anyoptimization/pymoo/blob/main/LICENSE>,
and <https://www.qt.io/development/open-source-lgpl-obligations>.
Do not assume PORTAL's MIT license removes dependency or combined-work
copyleft obligations.

PORTAL's NiaPy/pymoo integration modules are PORTAL-owned adapter code.
Their headers cite Vrbancic et al. (2018), JOSS 3(23), 613,
<https://doi.org/10.21105/joss.00613>, and Blank and Deb (2020),
IEEE Access 8, 89497-89509, <https://doi.org/10.1109/ACCESS.2020.2990567>.
They do not bundle those packages' optimizer implementations.
