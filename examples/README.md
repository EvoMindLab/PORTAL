# Small execution examples

`recipes/` contains eight purpose-designed 2D fixtures in the user-defined
categories `Baseline` and `Transformed`. These are demonstration inputs, not
additional members of the released 348-recipe catalog or paper experiments.
Their generation settings are in `create_demo_recipes.py`.

From the repository root:

```bash
python run_suite_pipeline.py --config examples/suite_smoke.toml --dry-run
python run_suite_pipeline.py --config examples/suite_smoke.toml
python run_augmentation_dataset.py --config examples/augmentation_smoke.toml --dry-run
python run_augmentation_dataset.py --config examples/augmentation_smoke.toml
```

The suite example uses two workers, 16 optimizer runs of 400 FEs, and reduced
ELA sampling. It selects four recipes and two disjoint two-recipe collections.
The augmentation example generates four Type 3 instances with two workers.
Outputs stay under `suite_runs/` and `augmentation_runs/` and are Git-ignored.
Rerunning the commands exercises their normal resume behavior.

To rebuild the demonstration JSONs separately:

```bash
python examples/create_demo_recipes.py --output my_demo_recipes
```

Then point `run.recipes_root` in a copied configuration to that directory.
