"""Canonical performance names, legacy inputs, and third-party credit checks."""
from __future__ import annotations

import ast
from contextlib import closing
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Recipe_Selection._terminology import (
    normalize_performance_keys,
    performance_name,
    read_characterization_csv,
    resolve_performance_input,
)
from Recipe_Selection.suite_pipeline.config import load_config
from Recipe_Selection.suite_pipeline.cli import build_parser
from portal_studio.experiment_runner.performance import (
    PERFORMANCE_CHECKPOINTS_FE, checkpoint_best_so_far, median_log_error,
)
from portal_studio.experiment_runner.store import ResultStore
from portal_studio.experiment_runner.finished_archive import _snapshot


@pytest.mark.parametrize("before,after", [
    ("isa_weight", "performance_weight"), ("ISA_CHECKPOINTS_FE", "PERFORMANCE_CHECKPOINTS_FE"),
    ("nnd_isa", "nnd_performance"), ("delta_ISA", "delta_performance"),
    ("ISA", "Performance"), ("isa", "performance"),
    ("visualisation", "visualisation"), ("isalpha", "isalpha"),
    ("isadirectory", "isadirectory"),
])
def test_name_conversion_is_token_bounded(before, after):
    assert performance_name(before) == after


def test_legacy_configuration_normalizes_without_rewriting_paths(tmp_path):
    cfg = tmp_path / "legacy.toml"
    cfg.write_text('[isa]\nruns = 3\n[selection]\nisa_weight = 0.25\n', encoding="utf-8")
    before = cfg.read_bytes()
    actual = load_config(cfg, ROOT)
    assert actual["performance"]["runs"] == 3
    assert actual["selection"]["performance_weight"] == 0.25
    assert "isa" not in actual and "isa_weight" not in actual["selection"]
    assert cfg.read_bytes() == before
    assert normalize_performance_keys({"isa_cleaned": "pipeline_output/isa_cleaned.csv"}) == {
        "performance_cleaned": "pipeline_output/isa_cleaned.csv",
    }


def test_conflicting_legacy_configuration_fails_explicitly(tmp_path):
    cfg = tmp_path / "conflict.toml"
    cfg.write_text('[isa]\nruns = 3\n[performance]\nruns = 5\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting legacy/canonical"):
        load_config(cfg, ROOT)


def test_legacy_stage_token_is_only_an_alias():
    parser = build_parser()
    args = parser.parse_args(["--config", "unused.toml", "--from-stage", "isa",
                              "--force-stage", "isa"])
    assert args.from_stage == "performance" and args.force_stage == ["performance"]
    assert "prepare" in parser.format_help() and "performance" in parser.format_help()


def test_legacy_csv_is_read_canonically_without_being_modified(tmp_path):
    legacy = tmp_path / "isa_cleaned.csv"
    legacy.write_text("recipe_id,nnd_isa,space\na,0.125,ISA\n", encoding="utf-8")
    before = legacy.read_bytes()
    requested = tmp_path / "performance_cleaned.csv"
    actual = read_characterization_csv(requested)
    assert actual.loc[0, "nnd_performance"] == 0.125
    assert actual.loc[0, "space"] == "Performance"
    assert legacy.read_bytes() == before and not requested.exists()


def test_canonical_csv_is_preferred_when_both_exist(tmp_path):
    (tmp_path / "isa_cleaned.csv").write_text("recipe_id,value\nold,1\n", encoding="utf-8")
    canonical = tmp_path / "performance_cleaned.csv"
    canonical.write_text("recipe_id,value\nnew,2\n", encoding="utf-8")
    assert resolve_performance_input(canonical) == canonical
    assert read_characterization_csv(canonical).loc[0, "recipe_id"] == "new"


def test_legacy_directory_and_filename_can_use_different_case(tmp_path):
    old_dir = tmp_path / "PORTAL_raw_ELA_ISA_required"
    old_dir.mkdir()
    old_file = old_dir / "isa_raw.csv"
    old_file.write_text("recipe_id,value\na,1\n", encoding="utf-8")
    requested = tmp_path / "PORTAL_raw_ELA_performance_required" / "performance_raw.csv"
    assert resolve_performance_input(requested) == old_file


def test_conflicting_legacy_columns_are_not_silently_combined(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("nnd_isa,nnd_performance\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns would collide"):
        read_characterization_csv(source)


def _make_legacy_database(path, *, canonical=False, conflict=False):
    with closing(sqlite3.connect(path)) as cx, cx:
        cx.execute("CREATE TABLE isa_checkpoints (run_uid TEXT, runset_uid TEXT, "
                   "checkpoint_fe INTEGER, absolute_error REAL, PRIMARY KEY(run_uid,checkpoint_fe))")
        cx.executemany("INSERT INTO isa_checkpoints VALUES (?,?,?,?)",
                       [("r1", "keep", 100, 2.0), ("r2", "discard", 100, 3.0)])
        if canonical:
            cx.execute("CREATE TABLE performance_checkpoints (run_uid TEXT, runset_uid TEXT, "
                       "checkpoint_fe INTEGER, absolute_error REAL, PRIMARY KEY(run_uid,checkpoint_fe))")
            cx.execute("INSERT INTO performance_checkpoints VALUES (?,?,?,?)",
                       ("r1", "keep", 100, 9.0 if conflict else 2.0))


@pytest.mark.parametrize("canonical", [False, True])
def test_writable_legacy_database_migrates_without_row_loss(tmp_path, canonical):
    source = tmp_path / "summary.sqlite"
    _make_legacy_database(source, canonical=canonical)
    store = ResultStore(tmp_path)
    store.ensure_schema()
    store.ensure_schema()  # Idempotent on the canonical schema.
    with closing(sqlite3.connect(source)) as cx:
        assert cx.execute("SELECT * FROM performance_checkpoints ORDER BY run_uid").fetchall() == [
            ("r1", "keep", 100, 2.0), ("r2", "discard", 100, 3.0),
        ]
        assert not cx.execute("SELECT 1 FROM sqlite_master WHERE name='isa_checkpoints'").fetchone()


def test_conflicting_checkpoint_tables_are_preserved(tmp_path):
    source = tmp_path / "summary.sqlite"
    _make_legacy_database(source, canonical=True, conflict=True)
    with pytest.raises(ValueError, match="both tables were retained"):
        ResultStore(tmp_path).ensure_schema()
    with closing(sqlite3.connect(source)) as cx:
        assert cx.execute("SELECT COUNT(*) FROM isa_checkpoints").fetchone()[0] == 2
        assert cx.execute("SELECT absolute_error FROM performance_checkpoints").fetchone()[0] == 9.0


def test_archiving_legacy_checkpoints_only_migrates_the_copy(tmp_path):
    source, destination = tmp_path / "old.sqlite", tmp_path / "copy.sqlite"
    _make_legacy_database(source)
    with closing(sqlite3.connect(source)) as cx, cx:
        cx.execute("CREATE TABLE runs (run_uid TEXT, runset_uid TEXT)")
        cx.execute("CREATE TABLE runsets (runset_uid TEXT)")
        cx.executemany("INSERT INTO runs VALUES (?,?)", [("r1", "keep"), ("r2", "discard")])
        cx.executemany("INSERT INTO runsets VALUES (?)", [("keep",), ("discard",)])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    _snapshot(source, destination, ["keep"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with closing(sqlite3.connect(destination)) as cx:
        assert cx.execute("SELECT * FROM performance_checkpoints").fetchall() == [("r1", "keep", 100, 2.0)]
        assert cx.execute("SELECT * FROM runs").fetchall() == [("r1", "keep")]


def test_legacy_module_alias_has_identical_checkpoint_math():
    legacy = importlib.import_module("portal_studio.experiment_runner.isa")
    assert legacy.ISA_CHECKPOINTS_FE == PERFORMANCE_CHECKPOINTS_FE
    assert legacy.checkpoint_best_so_far is checkpoint_best_so_far
    assert median_log_error([1.0, 2.0, 3.0]) == pytest.approx(np.median(np.log(np.array([1., 2., 3.]) + 1e-10)))


def test_legacy_studio_command_routes_to_performance(monkeypatch):
    from portal_studio import __main__ as cli
    calls = []
    monkeypatch.setattr(cli, "_run_script", lambda name, extra: calls.append((name, extra)) or 0)
    assert cli.main(["isa"]) == 0
    assert cli.main(["performance"]) == 0
    assert calls == [("run_curation_solvers.py", []), ("run_curation_solvers.py", [])]


@pytest.mark.parametrize("name", ["__init__", "classical_ela_features", "misc_features",
                                   "sampling", "utils", "local_optima_network_features"])
def test_vendored_feature_modules_credit_the_original_source(name):
    doc = ast.get_docstring(ast.parse((ROOT / f"pflacco/{name}.py").read_text(encoding="utf-8-sig")))
    assert "Prager" in doc and "https://github.com/Reiyan/pflacco" in doc
    assert "10.1162/evco_a_00341" in doc and "MIT" in doc


def test_ela_integration_layers_have_source_citations():
    for name in ("Recipe_Selection/run_ela.py", "portal_studio/experiment/ela.py"):
        doc = ast.get_docstring(ast.parse((ROOT / name).read_text(encoding="utf-8-sig")))
        assert "https://github.com/Reiyan/pflacco" in doc and "10.1162/evco_a_00341" in doc
