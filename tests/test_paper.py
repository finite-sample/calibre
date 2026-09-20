"""Check the manuscript's numerical evidence and executable example."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from benchmarks import aggregate, config, paper, run


@pytest.fixture
def rows():
    return paper.read_rows(paper.RESULTS / "raw.csv")


def test_complete_grid(rows):
    paper.validate_grid(rows)


@pytest.mark.parametrize("damage", ["duplicate", "missing_seed", "missing_method"])
def test_paper_refuses_incomplete_evidence(rows, damage):
    if damage == "duplicate":
        rows.append(rows[0].copy())
    elif damage == "missing_seed":
        rows.pop()
    else:
        rows = [r for r in rows if r["method"] != "calibre_spline"]
    with pytest.raises(ValueError, match="complete, unique"):
        paper.validate_grid(rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("brier", "nan", "Non-finite"),
        ("brier", "0.99", "decomposition"),
        ("auc", "1.1", "AUC"),
        ("n_distinct", "100000", "Distinct"),
    ],
)
def test_paper_rejects_invalid_measurements(rows, field, value, message):
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        paper.validate_grid(rows)


def test_generated_numbers_are_computed_from_evidence(rows, tmp_path):
    summary = aggregate.summarize(rows)
    paired = aggregate.pair_against_baseline(rows, config.BASELINE, 100)
    paper.write_numbers(rows, summary, paired, tmp_path)
    numbers = dict(
        re.findall(
            r"\\newcommand\{\\(\w+)\}\{([^}]+)\}",
            (tmp_path / "numbers.tex").read_text(),
        )
    )
    selected = [
        float(r["brier"])
        for r in rows
        if r["dataset"] == "overconfident" and r["method"] == "calibre_centered"
    ]
    assert float(numbers["CirBrier"]) == pytest.approx(np.mean(selected), abs=0.00005)
    assert int(numbers["BenchRows"]) == len(rows)
    paper.write_tables(summary, tmp_path)
    table = (tmp_path / "complete.tex").read_text()
    # All eight methods appear once in every dataset/model block.
    for name in paper.LABELS.values():
        assert sum(line.startswith(name + " &") for line in table.splitlines()) == 10
    assert "nan" not in table


def test_paired_plot_preserves_estimates_intervals_and_shared_scale(rows):
    paired = aggregate.pair_against_baseline(rows, config.BASELINE, 100)
    fig = paper.paired_figure(paired)
    cells = sorted({(r["dataset"], r["model"]) for r in paired})
    wanted = [m for m in paper.VISIBLE if m != config.BASELINE]
    indexed = {(r["dataset"], r["model"], r["method"]): r for r in paired}
    for ax, cell in zip(fig.axes, cells, strict=True):
        assert ax.get_xlim() == fig.axes[0].get_xlim()
        for i, method in enumerate(wanted):
            record = indexed[*cell, method]
            np.testing.assert_allclose(
                ax.lines[2 * i].get_xdata(),
                1000 * np.array([record["delta_brier_lo"], record["delta_brier_hi"]]),
            )
            np.testing.assert_allclose(
                ax.lines[2 * i + 1].get_xdata(), [1000 * record["delta_brier"]]
            )


def test_manuscript_example_uses_separate_observations(capsys):
    values = runpy.run_path(str(paper.ROOT / "ms" / "example.py"))
    assert values["report"].n_observations == 1000
    np.testing.assert_array_equal(
        values["predictions"], values["calibrator"].transform(values["scores"][1000:])
    )
    assert np.isfinite(values["report"].brier_score)
    assert capsys.readouterr().out


def test_artifact_build_and_manifest(tmp_path):
    assert paper.main(["--out", str(tmp_path)]) == 0
    for name in ("maps", "paired", "granularity"):
        assert (tmp_path / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["raw_sha256"]
        == hashlib.sha256((paper.RESULTS / "raw.csv").read_bytes()).hexdigest()
    )
    assert manifest["n_bootstrap"] == config.N_BOOTSTRAP


def test_provenance_hashes_actual_sources():
    env = run._environment()
    root = Path(__file__).resolve().parents[1]
    for path, digest in env["source_sha256"].items():
        assert digest == hashlib.sha256((root / path).read_bytes()).hexdigest()
    assert "calibre/calibrators/spline.py" in env["source_sha256"]
    assert "uv.lock" in env["source_sha256"]


def test_custom_run_keeps_provenance_next_to_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_cells", lambda args: [])
    monkeypatch.setattr(run, "_environment", lambda: {"test": True})
    out = tmp_path / "check.csv"
    assert run.main(["--out", str(out)]) == 0
    assert json.loads((tmp_path / "check.environment.json").read_text()) == {
        "test": True
    }
