"""Tests for the benchmark harness.

Cheap enough for CI: one small cell, plus the guards that keep the harness
honest. The full grid is run by hand and its results are committed.

The guards matter more than the happy path. A benchmark that silently drops a
failing configuration, or that lets calibre's isotonic wrapper drift away from the
scikit-learn baseline it is measured against, would report a flattering number
without anyone noticing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from benchmarks import (
    aggregate,
    config,
    datasets,
    figures,
    measures,
    methods,
    models,
    protocol,
    run,
)


def test_decomposition_figure_accepts_committed_benchmark_schema(monkeypatch):
    """The benchmark's compact column names must reach the public plotting API."""
    summary = [
        {
            "dataset": "example",
            "model": "identity",
            "method": method,
            "brier": brier,
            "mcb": mcb,
            "dsc": dsc,
        }
        for method, brier, mcb, dsc in (
            ("uncalibrated", "0.20", "0.03", "0.08"),
            ("calibre_isotonic", "0.18", "0.01", "0.07"),
        )
    ]
    monkeypatch.setattr(figures, "_save", lambda figure, stem: None)

    figures.decomposition(summary, "example", "identity")


@pytest.mark.parametrize(
    ("status", "expected"),
    [("", "abc1234"), (" M README.md\n", "abc1234-dirty")],
)
def test_benchmark_git_sha_records_dirty_state(monkeypatch, status, expected):
    """Committed results must say when uncommitted code produced them."""
    outputs = iter(["abc1234\n", status])

    def fake_run(*args, **kwargs):
        return type("Result", (), {"stdout": next(outputs)})()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run._git_sha() == expected


def test_the_quick_configuration_is_offline():
    """`--quick` must not need the network, or CI cannot run it."""
    for name in config.QUICK_DATASETS:
        assert name not in config.REMOTE_DATASETS, (
            f"{name} needs a network fetch but is in the quick set"
        )


def test_committed_quick_artifact_uses_current_default_methods():
    """A removed method must not survive in the reproducibility artifact."""
    path = Path(__file__).resolve().parents[1] / "benchmarks/results/quick.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert {row["method"] for row in rows} == set(methods.available())
    aggregate.check_completeness(rows, len(config.QUICK_SEEDS))


def test_every_method_name_has_a_family():
    """The figures group by family, so a method without one would vanish."""
    for name in methods.available():
        assert name in methods.METHODS


def test_the_primary_metrics_are_columns_the_harness_records():
    """`headline_table` ranks by `PRIMARY_METRICS[0]`, so it has to exist.

    Declaring the headline metric before the grid runs is what stops it being
    chosen after the numbers are in, and `figures.headline_table` reads the
    declaration rather than naming a column itself. That leaves one way to break
    it: rename a measure and leave the declaration behind, at which point the
    table would raise a `KeyError` on every row during a docs build.

    This does not execute `headline_table` — that writes `headline.csv` into
    `docs/`, and a test must not edit the repository.
    """
    missing = [m for m in config.PRIMARY_METRICS if m not in measures.COLUMNS]
    assert not missing, f"declared as primary but never measured: {missing}"


def test_calibre_methods_are_built_at_library_defaults():
    """`CALIBRATOR_DEFAULTS_ONLY` is the rule; this is what keeps it.

    Tuning calibre's calibrators against an untuned scikit-learn baseline would
    decide the comparison by construction rather than by measurement. Until now
    the flag only recorded the intention in prose — nothing would have noticed a
    hyperparameter appearing in `methods._build`, and the benchmark would have
    gone on reporting a number the README presents as a fair fight.

    Comparing each built calibrator against a bare instance of its own class is
    the check: identical today because `_build` passes no arguments, and that is
    exactly the property at risk.
    """
    if not config.CALIBRATOR_DEFAULTS_ONLY:
        pytest.skip("CALIBRATOR_DEFAULTS_ONLY is off; tuning is permitted")
    for name in methods.METHODS:
        if not name.startswith("calibre_"):
            continue
        built = methods._build(name)
        assert built.get_params() == type(built)().get_params(), (
            f"{name} is not constructed at library defaults"
        )


def test_a_cell_produces_one_row_per_method_with_every_column():
    """The schema `aggregate.py` depends on."""
    names = ["uncalibrated", "sklearn_isotonic", "calibre_isotonic", "calibre_centered"]
    rows = protocol.run_cell("overconfident", "identity", 0, names)

    assert [r["method"] for r in rows] == names
    for row in rows:
        missing = set(measures.COLUMNS) - set(row)
        assert missing == set(), f"{row['method']} is missing {sorted(missing)}"


def test_the_isotonic_self_check_is_live():
    """calibre's wrapper must reproduce the baseline it is measured against.

    Asserted here as well as inside ``run_cell`` so that a change silencing the
    self-check shows up as a test failure rather than as a quietly better score.
    """
    names = ["sklearn_isotonic", "calibre_isotonic"]
    rows = protocol.run_cell("overconfident", "identity", 1, names)
    by_method = {r["method"]: r for r in rows}
    assert by_method["calibre_isotonic"]["brier"] == pytest.approx(
        by_method["sklearn_isotonic"]["brier"], abs=1e-12
    )


def test_every_calibrator_sees_identical_inputs():
    """The fairness control, asserted rather than assumed.

    If the out-of-fold scores were recomputed per method, differences between
    calibrators would be confounded with resampling noise.
    """
    dataset = datasets.load("overconfident", 3)
    first = protocol._scores_for_cell(dataset, "identity", 3)
    second = protocol._scores_for_cell(dataset, "identity", 3)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_synthetic_cells_carry_the_known_truth():
    """`true_error` is the strongest evidence available and must be populated."""
    rows = protocol.run_cell("overconfident", "identity", 0, ["uncalibrated"])
    assert np.isfinite(rows[0]["true_error"])


def test_real_cells_report_no_true_error():
    """There is no truth to compare against on real data; it must not be invented."""
    rows = protocol.run_cell("breast_cancer", "logreg", 0, ["uncalibrated"])
    assert not np.isfinite(rows[0]["true_error"])


def test_completeness_guard_rejects_a_missing_seed():
    """A benchmark that quietly drops a failed cell flatters whatever survived."""
    rows = [
        {"dataset": "d", "model": "m", "method": "a", "seed": str(s)} for s in range(3)
    ]
    rows += [{"dataset": "d", "model": "m", "method": "b", "seed": "0"}]
    with pytest.raises(ValueError, match="do not have 3 seeds"):
        aggregate.check_completeness(rows, 3)


def test_completeness_guard_accepts_a_full_grid():
    """And it must not fire on a complete one."""
    rows = [
        {"dataset": "d", "model": "m", "method": method, "seed": str(seed)}
        for method in ("a", "b")
        for seed in range(3)
    ]
    aggregate.check_completeness(rows, 3)


def test_paired_comparison_is_signed_so_positive_means_better():
    """A sign error here would invert every claim on the docs page."""
    rows = [
        {
            "dataset": "d",
            "model": "m",
            "seed": "0",
            "method": "sklearn_isotonic",
            "brier": "0.20",
            "mcb": "0.02",
            "n_distinct": "50",
        },
        {
            "dataset": "d",
            "model": "m",
            "seed": "0",
            "method": "calibre_centered",
            "brier": "0.10",
            "mcb": "0.01",
            "n_distinct": "500",
        },
    ]
    paired = aggregate.pair_against_baseline(rows, "sklearn_isotonic", n_resamples=10)
    assert len(paired) == 1
    assert paired[0]["delta_brier"] == pytest.approx(0.10)
    assert paired[0]["distinct_ratio_vs_baseline"] == pytest.approx(10.0)


def test_netcal_is_absent_unless_requested():
    """An optional comparator must not creep into the default grid."""
    assert "netcal_beta" not in methods.available()
    assert "netcal_bbq" not in methods.available(include_netcal=True)


def test_models_build_without_fitting():
    """Each model name yields a pipeline, so a typo fails fast."""
    for name in models.MODELS:
        assert models.build(name, 0) is not None
    with pytest.raises(ValueError, match="unknown model"):
        models.build("nonexistent", 0)
