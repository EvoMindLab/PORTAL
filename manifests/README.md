# Recipe membership

These CSVs contain only identifiers, categories, and repository-relative JSON
paths. They contain no landscape features, optimizer measurements, or distances.

| File | Recipes |
|---|---:|
| `catalog.csv` | 348 |
| `retained_pool.csv` | 241 |
| `s50.csv` | 50 |
| `s20.csv` | 20 |
| `train30.csv` | 30 |
| `test30.csv` | 30 |

The retained pool is a subset of `Suite/`; it is not duplicated into another
241-file directory. The named suite folders contain byte-identical copies of
their catalog recipes. Training and test memberships are disjoint. Other suites
may share recipes; S20 is not a truncation of S50.
