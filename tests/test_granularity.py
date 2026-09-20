"""Decision identities and independent allocation checks for the experiment."""

import itertools

import numpy as np
import pytest

from experiments.granularity import study


def test_two_group_identity():
    for d in (0, 0.05, 0.15):
        p = np.array([0.3 - d, 0.3 + d])
        q = np.full(2, 0.3)
        row = study.losses(p, q)
        assert row["pooling_loss"] == pytest.approx(d * d)
        assert row["miscalibration"] == pytest.approx(0, abs=1e-15)
        thresholds = np.linspace(0, 1, 10001)
        curves = study.regret_curves(p, q, thresholds)
        assert 2 * np.trapezoid(curves[0], thresholds) == pytest.approx(d * d)
        assert 2 * np.trapezoid(curves[1], thresholds) == pytest.approx(d * d)


def test_decomposition_and_integrals_independent():
    p = np.array([0.1, 0.2, 0.4, 0.8])
    q = np.array([0.2, 0.2, 0.6, 0.6])
    r = np.array([0.15, 0.15, 0.6, 0.6])
    row = study.losses(p, q)
    np.testing.assert_allclose(study.conditional_risk(p, q), r)
    assert row["pooling_loss"] == pytest.approx(0.02125)
    assert row["total_error"] == pytest.approx(0.0225)
    assert row["total_error"] == pytest.approx(
        row["pooling_loss"] + row["miscalibration"]
    )
    # Midpoint integration avoids threshold-equality quadrature artifacts.
    thresholds = (np.arange(100000) + 0.5) / 100000
    curves = study.regret_curves(p, q, thresholds)
    np.testing.assert_allclose(2 * curves.mean(axis=1), [0.02125, 0.0225], atol=1e-12)
    assert row["expected_brier"] == pytest.approx(
        np.mean(p * (1 - q) ** 2 + (1 - p) * q**2)
    )


def test_information_and_noise_are_distinct():
    p = np.full(100, 0.3)
    q = np.linspace(0.2, 0.4, 100)
    row = study.losses(p, q)
    assert row["pooling_loss"] == 0
    assert row["total_error"] > 0
    p = np.linspace(0.1, 0.9, 100)
    assert study.losses(p, q)["pooling_loss"] == 0
    q = np.repeat([0.2, 0.8], 50)
    assert study.losses(p, q)["pooling_loss"] > 0
    assert study.losses(q, q)["pooling_loss"] == pytest.approx(0, abs=1e-15)


def test_pair_counts_and_undefined_correlation():
    row = study.ordering(np.array([0.3, 0.2, 0.2, 0.1]))
    assert row["pooled_pairs"] == pytest.approx(1 / 6)
    assert row["reversed_pairs"] == pytest.approx(5 / 6)
    assert np.isnan(study.ordering(np.zeros(4))["spearman"])
    assert study.ordering(np.arange(4))["spearman"] == pytest.approx(1)


def test_random_ties_match_exhaustive_lottery():
    p = np.array([0.1, 0.3, 0.6, 0.9])
    q = np.array([0.2, 0.5, 0.5, 0.5])
    actual = study.capacity_values(p, q, np.array([0, 0.5, 1]))
    expected = (
        np.mean([sum(p[list(pair)]) for pair in itertools.combinations([1, 2, 3], 2)])
        * 250
    )
    assert actual[0, 1] == pytest.approx(expected)
    assert actual[1, 1] == pytest.approx((0.6 + 0.9) * 250)
    np.testing.assert_allclose(actual[:, 0], 0)
    np.testing.assert_allclose(actual[:, 2], p.mean() * 1000)


def test_monotone_tiebreak_restores_original_allocation():
    s, p = study.population("linear")
    q = np.floor(s * 5) / 5
    np.testing.assert_allclose(
        study.capacity_values(p, q)[1], study.capacity_values(p, s)[1]
    )
    assert np.all(
        study.capacity_values(p, s)[1] >= study.capacity_values(p, q)[0] - 1e-10
    )


def test_oracle_capacity_dominates_and_nonmonotone_ranking_can_hurt():
    s, p = study.population("nonmonotone")
    oracle = study.capacity_values(p, p)[0]
    original = study.capacity_values(p, s)[0]
    constant = study.capacity_values(p, np.full(s.size, 0.5))[0]
    assert np.all(oracle >= original - 1e-10)
    assert np.any(original < constant)


def test_threshold_equality_and_rounding():
    p = np.array([0.1, 0.9])
    q = np.array([0.3, 0.3])
    curves = study.regret_curves(p, q, np.array([0, 0.3, 1]))
    assert curves[1, 1] == pytest.approx(0.3)
    np.testing.assert_allclose(curves[:, [0, 2]], 0, atol=1e-15)
    q = np.array([0.30000001, 0.30000002])
    assert study.losses(p, q)["pooling_loss"] == 0
    assert study.losses(p, np.round(q, 6))["pooling_loss"] == pytest.approx(0.16)


@pytest.mark.parametrize(
    ("p", "q"),
    [
        ([0.1], [0.2]),
        ([0.1, 0.2], [0.3]),
        ([0.1, np.nan], [0.1, 0.2]),
        ([-0.1, 0.1], [0.1, 0.2]),
    ],
)
def test_invalid_populations_rejected(p, q):
    with pytest.raises(ValueError, match=r"required|finite|lie in"):
        study.losses(np.array(p), np.array(q))


def test_population_designs_and_grid_guard():
    for design in study.DESIGNS:
        s, p = study.population(design)
        assert len(s) == 1000
        assert np.all(np.diff(s) > 0)
        assert np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError, match="incomplete"):
        study.evaluate(np.zeros((1, 1)))


def test_all_method_pilot():
    cell, fitted = study.run_cell(("linear", 200, 0))
    assert cell == ("linear", 200, 0)
    assert fitted.shape == (7, 1000)
    assert np.all(np.isfinite(fitted))
    for q in fitted:
        assert study.losses(study.population("linear")[1], q)["pooling_loss"] >= 0


def test_rounding_only_coarsens_and_information_regret_is_lower_bound():
    rng = np.random.default_rng(71)
    p = rng.uniform(0, 1, 20)
    q = rng.uniform(0, 1, 20)
    rounded = np.round(q, 1)
    assert (
        study.losses(p, rounded)["pooling_loss"] >= study.losses(p, q)["pooling_loss"]
    )
    curves = study.regret_curves(p, rounded)
    assert np.all(curves[1] >= curves[0] - 1e-12)


def test_interval_uses_paired_seed_differences():
    from experiments.granularity.report import bootstrap_weights, interval

    baseline = np.arange(30, dtype=float)
    alternative = baseline + 0.125
    weights = bootstrap_weights()
    mean, low, high = interval(alternative - baseline, weights)
    np.testing.assert_allclose([mean, low, high], 0.125)
    np.testing.assert_allclose(weights.sum(axis=1), 1)


def test_regret_figure_plots_paired_values_and_common_axes():
    from experiments.granularity import report

    curves = np.zeros((5, 3, 30, 7, 2, 2, 501))
    curves[0, 1, :, 2, 0, 1, :] = 0.002
    with report.plt.rc_context(report.THEME):
        fig = report.threshold_figure(curves, 0, 0, report.bootstrap_weights())
    right = fig.axes[1]
    line = next(line for line in right.lines if line.get_label() == "Centered isotonic")
    np.testing.assert_allclose(line.get_ydata(), 2)
    assert fig.axes[0].get_ylim() == right.get_ylim()
    report.plt.close(fig)


def test_result_loader_rejects_wrong_configuration(tmp_path):
    import json

    from experiments.granularity.report import load_results

    (tmp_path / "manifest.json").write_text(json.dumps({"configuration": {}}))
    with pytest.raises(ValueError, match="configuration"):
        load_results(tmp_path)


def test_evaluation_reuse_rejects_changed_configuration(tmp_path, monkeypatch):
    import json
    import sys

    (tmp_path / "config.json").write_text(json.dumps({"seeds": [0]}))
    monkeypatch.setattr(
        sys, "argv", ["study", "--out", str(tmp_path), "--evaluate-only"]
    )
    with pytest.raises(ValueError, match="saved configuration differs"):
        study.main()


def test_replacement_loss_equals_conditional_boundary_crossing():
    p = np.array([0.1, 0.3, 0.3, 0.7, 0.9, 0.9])
    q = np.repeat([0.2, 0.5, 0.8], 2)
    thresholds = np.unique(np.r_[0, p, (p[:-1] + p[1:]) / 2, 1])
    expected = np.zeros(thresholds.size)
    crossing = np.zeros(thresholds.size, dtype=bool)
    for value in np.unique(q):
        group = p[q == value]
        positive = np.maximum(group[:, None] - thresholds, 0).mean(axis=0)
        negative = np.maximum(thresholds - group[:, None], 0).mean(axis=0)
        expected += (group.size / p.size) * np.minimum(positive, negative)
        crossing |= (group.min() < thresholds) & (thresholds < group.max())
    actual = study.regret_curves(p, q, thresholds)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_array_equal(actual > 1e-14, crossing)


def test_report_can_preserve_threshold_decisions_without_preserving_risks():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    q = np.array([0.15, 0.15, 0.85, 0.85])
    assert study.losses(p, q)["pooling_loss"] > 0
    np.testing.assert_allclose(
        study.regret_curves(p, q, np.array([0.3, 0.5, 0.7])), 0, atol=1e-15
    )
    assert study.regret_curves(p, q, np.array([0.15]))[0, 0] > 0


def test_literal_regret_is_risk_weighted_action_disagreement():
    p = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    q = np.array([0.0, 0.9, 0.1, 0.8, 1.0])
    thresholds = np.unique(np.r_[p, q, 0.3, 0.6])
    expected = (
        np.abs(p[:, None] - thresholds)
        * ((q[:, None] > thresholds) != (p[:, None] > thresholds))
    ).mean(axis=0)
    actual = study.regret_curves(p, q, thresholds)[1]
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_invertible_reversed_report_has_information_but_bad_literal_actions():
    p = np.array([0.1, 0.3, 0.7, 0.9])
    q = 1 - p
    curves = study.regret_curves(p, q, np.array([0.5]))
    assert curves[0, 0] == pytest.approx(0)
    assert curves[1, 0] > 0
    decoded = 1 - q
    np.testing.assert_allclose(study.regret_curves(p, decoded), 0, atol=1e-15)
