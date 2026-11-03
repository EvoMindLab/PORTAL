"""Check that PORTAL's MIT grant does not overwrite third-party terms."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_root_mit_license_names_portal_contributors():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n")
    assert "Copyright (c) 2025-2026 Danial Yazdani and PORTAL contributors" in text
    assert "Permission is hereby granted, free of charge" in text


def test_optimizer_terms_are_separate_from_root_mit():
    assert "GNU GENERAL PUBLIC LICENSE" in (ROOT / "algorithms/LICENSE").read_text()
    scope = (ROOT / "LICENSING.md").read_text()
    assert "outside the root MIT grant" in scope
    assert "base_algorithm.py" in scope
    assert "does not supply" in scope and "missing permission" in scope


def test_pflacco_preserves_original_copyright_and_permission():
    text = (ROOT / "pflacco/LICENSE").read_text()
    assert "Copyright (c) 2019 Raphael Patrick Prager" in text
    assert "Danial Yazdani" not in text
    assert "Permission is hereby granted" in text


def test_optimizer_inventory_covers_every_implementation():
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    paths = [item["path"] for item in data["algorithms"]]
    expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / "algorithms").glob("*.py")
                if path.name != "base_algorithm.py"}
    assert len(paths) == len(set(paths)) and set(paths) == expected
    assert all(item["local_contribution_license"] == "GPL-3.0-only" for item in data["algorithms"])
    assert all(item["release_clearance"] in {"documented", "permission-required", "provenance-review"}
               for item in data["algorithms"])


def test_uncleared_upstream_code_is_not_silently_declared_mit():
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    for item in data["algorithms"]:
        if item["release_clearance"] == "permission-required":
            assert item["upstream_license"] == "UNVERIFIED"
            assert item["provenance"] == "corroborated-author-source-reuse"
    text = (ROOT / "README.md").read_text()
    assert "does **not** relicense" in text and "excluded from the public release" in text


@pytest.mark.parametrize("name", ["benchmark_generator", "compose_blocks", "portal_default_settings"])
def test_portal_core_headers_use_mit_without_removing_authors(name):
    text = (ROOT / f"portal_core/{name}.py").read_text(encoding="utf-8-sig")
    doc = ast.get_docstring(ast.parse(text))
    assert "MIT for PORTAL-owned code" in doc
    assert "GNU General Public License" not in doc
    assert all(author in doc for author in ("Danial Yazdani", "Delaram Yazdani", "Mai Peng"))


def test_software_citation_qualifies_the_mit_scope():
    text = (ROOT / "CITATION.cff").read_text()
    assert "license: MIT" in text
    assert "PORTAL-owned framework material only" in text
    assert "given-names: Danial" in text and "family-names: Yazdani" in text


@pytest.mark.parametrize("module", ["SHADE.py", "LSHADE.py"])
def test_preserved_tanabe_license_and_source_credits(module):
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    item = next(entry for entry in data["algorithms"] if entry["path"] == "algorithms/" + module)
    assert item["release_clearance"] == "documented"
    assert item["upstream_license"].startswith("GPL-3.0")
    license_bytes = (ROOT / item["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == "589ed823e9a84c56feb95ac58e7cf384626b9cbf4fda2a907bc36e103de1bad2"
    notice = (ROOT / item["notice_file"]).read_text()
    assert "Ryoji Tanabe" in notice and "Jingqiao Zhang" in notice


def test_confirmed_paper_implementations_record_the_maintainer_basis():
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    confirmed = [item for item in data["algorithms"]
                 if item["provenance"] == "maintainer-confirmed-paper-implementation"]
    assert len(confirmed) == 10
    for item in confirmed:
        assert item["release_clearance"] == "documented"
        assert item["maintainer_confirmation"]["confirmed_by"] == "Danial Yazdani"
        assert item["maintainer_confirmation"]["date"] == "2026-11-03"
        assert "not independently verified authorship" in item["review_note"]
        assert item["source_url"] and "required_confirmation" not in item


def test_dd_cma_exclusion_is_explicit_and_not_a_license_grant():
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    excluded = {item["path"] for item in data["public_release_exclusions"]}
    assert excluded == {"algorithms/dd-CMA-ES.py"}
    entries = {item["path"]: item for item in data["algorithms"]}
    if "algorithms/dd-CMA-ES.py" in entries:  # Canonical private research tree.
        assert entries["algorithms/dd-CMA-ES.py"]["release_clearance"] == "permission-required"
    else:  # Distributed public tree.
        assert not (ROOT / "algorithms/dd-CMA-ES.py").exists()
        assert all(item["release_clearance"] == "documented" for item in entries.values())


def test_reviewed_headers_do_not_overstate_translation_provenance():
    data = json.loads((ROOT / "algorithms/license_inventory.json").read_text())
    for item in data["algorithms"]:
        if item["provenance"] in {"source-informed-reimplementation",
                                  "maintainer-confirmed-paper-implementation"}:
            doc = ast.get_docstring(ast.parse((ROOT / item["path"]).read_text(encoding="utf-8-sig")))
            assert "direct Python translation" not in doc
            assert "algorithms/PROVENANCE.md" in doc
