"""Self-contained checks for the source-tree public distribution.

No private measurements, manuscript snapshots, or long experiments are used.
Run with: python -m pytest tests/test_public_release.py -q
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from portal_core.fitness import fitness, fitness_batch
from portal_core.load_instance import load_instance
from portal_core.realization import realize
from portal_core.save_instance import save_instance
from Recipe_Selection.suite_pipeline.discovery import discover_recipes


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_version_metadata_and_command_line_agree(capsys):
    import re
    import yaml
    from portal_studio import __version__
    from portal_studio.__main__ import main

    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["version"] == __version__
    assert f"## [{__version__}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"**v{__version__}**" in (ROOT / "README.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f"PORTAL {__version__}"


def test_catalog_loads_and_has_expected_structure():
    records = discover_recipes(ROOT / "Suite", require_single_dimension=True)
    assert len(records) == 348
    assert Counter(record.family for record in records) == {
        "Family1": 35, "Family2": 103, "Family3": 112, "Family4": 98,
    }
    for record in records:
        instance = load_instance(str(record.path))
        assert instance["d"] == 30
        x = np.linspace(-35.0, 45.0, instance["d"])
        assert np.isfinite(float(fitness(x, instance)))


def test_public_memberships_are_consistent():
    manifests = ROOT / "manifests"
    if not manifests.exists():
        pytest.skip("Membership manifests are materialized by the release exporter.")
    members = {}
    for name, expected in (("catalog", 348), ("retained_pool", 241), ("s50", 50),
                           ("s20", 20), ("train30", 30), ("test30", 30)):
        with (manifests / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ["recipe_id", "family", "path"]
            rows = list(reader)
        ids = {row["recipe_id"] for row in rows}
        assert len(rows) == len(ids) == expected
        members[name] = ids
        for row in rows:
            released = ROOT / row["path"]
            catalog = ROOT / "Suite" / row["family"] / "d30" / f"{row['recipe_id']}.json"
            assert digest(released) == digest(catalog)
    assert not members["train30"] & members["test30"]
    assert len(members["s20"] & members["s50"]) == 13
    for name in ("s50", "s20", "train30", "test30"):
        assert members[name] <= members["retained_pool"]


@pytest.mark.parametrize("family", [1, 2, 3, 4])
def test_unified_round_trip_and_batch_evaluation(tmp_path, family):
    source = ROOT / "Suite" / f"Family{family}" / "d30" / f"f0{family}_r001.json"
    instance = load_instance(str(source))
    samples = np.random.default_rng(4).uniform(-40, 40, size=(3, 30))
    scalar = np.array([float(fitness(x, instance)) for x in samples])
    batched = fitness_batch(samples, instance, evaluation_device="numpy")
    np.testing.assert_allclose(batched, scalar, rtol=1e-10, atol=1e-9)
    path = save_instance(instance, filename="roundtrip.json", output_dir=str(tmp_path))
    restored = load_instance(path)
    actual = [float(fitness(x, restored)) for x in samples]
    np.testing.assert_allclose(actual, scalar, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("family", [1, 4])
@pytest.mark.parametrize("method", ["center_beta", "diverse", "augment"])
def test_realization_modes(tmp_path, family, method):
    source = ROOT / "Suite" / f"Family{family}" / "d30" / f"f0{family}_r001.json"
    before = digest(source)
    instance = load_instance(str(source))
    result = realize(instance, 42, 1, method=method, family_id=family, recipe_index=1)
    path = save_instance(result, filename="realized.json", output_dir=str(tmp_path))
    loaded = load_instance(path)
    assert loaded["d"] == instance["d"]
    assert np.isfinite(float(fitness(np.zeros(loaded["d"]), loaded)))
    assert digest(source) == before


def test_native_algorithm_dependencies_are_complete():
    from portal_studio.algorithms.registry import DISCOVERY_SKIPPED, SOLVERS
    assert not DISCOVERY_SKIPPED
    assert "IPOP-CMA-ES" in SOLVERS


@pytest.mark.parametrize("name", ["CMA-ES", "DE", "LSHADE", "PSO", "CLPSO",
                                  "PatternSearch", "Powell", "MA-ES", "TAPSO", "L-SRTDE",
                                  "IPOP-CMA-ES"])
def test_native_optimizer_execution(name):
    from portal_studio.algorithms.runner import run_solver_on_instance
    instance = load_instance(str(ROOT / "examples/recipes/demo_02.json"))
    result = run_solver_on_instance(instance=instance, solver_name=name, budget=200, seed=42)
    assert 0 < result.problem.fe_count <= 200
    assert np.isfinite(result.problem.best_f)


def test_gui_workspaces_and_ela_export(tmp_path):
    from PySide6.QtWidgets import QApplication
    from portal_studio.gui.main_window import PortalStudioMainWindow
    from portal_studio.experiment.ela import compute_ela, save_ela_xlsx
    import pandas as pd
    app = QApplication.instance() or QApplication([])
    window = PortalStudioMainWindow()
    from portal_studio import __version__
    assert window.windowTitle() == f"PORTAL Studio {__version__}"
    window.show()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Generator", "Experiment", "Education",
    ]
    for i in range(3):
        window.tabs.setCurrentIndex(i)
        app.processEvents()
    source = ROOT / "examples/recipes/demo_02.json"
    instance = load_instance(str(source))
    result = compute_ela(instance, groups=["ela_meta", "pca"], sample_factor=30, seed=42)
    assert not result.failed_groups
    frame = pd.DataFrame([result.features])
    output = save_ela_xlsx(frame, tmp_path / "features.xlsx")
    assert Path(output).stat().st_size > 0
    window.close()
    app.processEvents()


def test_no_private_artifacts_in_public_release():
    if not (ROOT / "manifests").exists():
        pytest.skip("Export-only distribution check.")
    for folder in ("MATLAB", "Python", "Stress_Test_Study", "Realization_Study",
                   "S20_Algorithm_Study", "Selection_Sensitivity_Study",
                   "PORTAL_raw_ELA_performance_required_20260524", "Recipe_Catalog_Table"):
        assert not (ROOT / folder).exists()
    assert not list((ROOT / "Suite").rglob("*.jsonl"))
    assert not (ROOT / "Documentation/Part2_main.tex").exists()
