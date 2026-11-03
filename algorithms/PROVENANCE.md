# Optimizer provenance and release scope

Reviewed November 3, 2026. The optimizer methods are credited to their original
researchers; PORTAL does not claim to have invented them. The root MIT license
covers PORTAL-owned framework material, not these optimizer modules. See
[licensing scope](../LICENSING.md) and [optimizer terms](README.md).

## What the classifications mean

A published algorithm is not the same thing as its source-code implementation.
An independently written implementation of equations/pseudocode does not
automatically inherit another implementation's license. Conversely, translating
or adapting protected source expression can require an applicable grant.
Looking at code for numerical cross-checking does not, by itself, establish
copying.

## Maintainer-confirmed paper implementations

Danial Yazdani confirmed that the following ten implementations were developed
from their papers, with author reference code consulted for clarification.
This records the maintainer's account of development history, not a claim that
a source comparison independently proved authorship. The earlier permission
holds and direct-translation descriptions for these ten are superseded by this
confirmation. Scientific citations, source URLs, and numerical behavior remain
unchanged. Their local implementation code retains the existing GPL v3 terms.

## Results for the 13 previously reviewed modules

| Module | Evidence and current outcome |
|---|---|
| `SHADE.py` | Exact SHADE 1.1.1 MATLAB/Octave archive located. Author repository declares GPL v3; license and source credits preserved locally. Documented under GPL, not MIT. |
| `LSHADE.py` | Exact L-SHADE 1.0.1 MATLAB/Octave archive located. Same GPL repository declaration and preserved notices. Documented under GPL, not MIT. |
| `dd-CMA-ES.py` | Excluded from the public release by maintainer decision: author-core reuse was corroborated, but no explicit license grant was found. |
| `CLPSO.py` | Maintainer-confirmed paper implementation; reference `CLPSO_new_func.m` consulted for clarification. |
| `HCLPSO.py` | Maintainer-confirmed paper implementation; reference `HCLPSO_PS_15_35.m` consulted for clarification. |
| `SaDE.py` | Maintainer-confirmed paper implementation; reference `SaDE.m` consulted for clarification. |
| `GLPSO.py` | Maintainer-confirmed paper implementation; reference `glpso.cpp` consulted for clarification. |
| `EA4eig.py` | Maintainer-confirmed paper implementation; CEC2022 MATLAB references consulted for clarification. |
| `IMODE.py` | Maintainer-confirmed paper implementation; CEC2020 MATLAB references consulted for clarification. |
| `jSO.py` | Maintainer-confirmed paper implementation; CEC2017 C++ references consulted for clarification. |
| `LSHADE-cnEpSin.py` | Maintainer-confirmed paper implementation; CEC2017 MATLAB reference consulted for clarification. |
| `LSHADE-RSP.py` | Maintainer-confirmed paper implementation; author `main.cpp` consulted for clarification. |
| `NL-SHADE-RSP.py` | Maintainer-confirmed paper implementation; CEC2021 C++ reference consulted for clarification. |

The per-file inventory retains the exact source URLs and records who confirmed
the development history and when. It does not assign a new license to those
authors' separate reference code.

## Preserved SHADE / L-SHADE license and credits

The exact archives use dots, not underscores, in their version names:

- <https://ryojitanabe.github.io/code/SHADE1.1.1_CEC2014_Matlab_Octave.zip>
- <https://ryojitanabe.github.io/code/LSHADE1.0.1_CEC2014_Octave-Matlab.zip>

They are published in Ryoji Tanabe's repository, whose checked revision
`9aa07eea380ef3a437b3ba1a9ffdbfc4dac58a64` declares GPL v3 at its root:
<https://github.com/ryojitanabe/ryojitanabe.github.io/blob/9aa07eea380ef3a437b3ba1a9ffdbfc4dac58a64/LICENSE>.
No separate conflicting archive license was found. This is repository-level
license evidence, not an invented archive-specific or MIT grant.

The exact license is saved in
[LICENSES/Tanabe-GPL-3.0.txt](../LICENSES/Tanabe-GPL-3.0.txt).
The associated Tanabe/Fukunaga and Jingqiao Zhang utility credits are preserved
in [LICENSES/Tanabe-NOTICE.txt](../LICENSES/Tanabe-NOTICE.txt).

## Excluded author-derived implementation

The maintainer approved excluding `algorithms/dd-CMA-ES.py` from the public
release. Its numerical core substantially matches the author source. Detailed
comparison evidence is retained privately, not distributed as research data.

Author source:
<https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518>.
The checked author-lab copy at
<https://github.com/akimotolab/multi-fidelity/blob/master/ddcma.py> also refers
back to that gist. Neither checked source supplied an explicit license grant.
Do not substitute an unrelated reimplementation's license for the author's
permission.

The file remains in the private research checkout only. It was not used in the
reported studies and was already excluded from Studio's registered optimizers.
Its exclusion does not change the included algorithms or scientific results.

## Release handling

The [inventory](license_inventory.json) records per-module sources, license
scope, and review basis. The public inventory lists only included modules and
records the approved exclusion separately. Unknown provenance or missing
permission still blocks export for any module not explicitly excluded.

Original scientific references and source credits remain in the modules.
PORTAL's root MIT license and the separate local optimizer GPL terms are
unchanged by this clarification.
