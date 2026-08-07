"""Pin ``smooth_calibration_error`` against Apple's ``relplot``.

calibre reimplements smECE rather than depending on ``relplot``, which pulls in
seaborn and matplotlib. A reimplementation is only worth having if it is exact,
so it is checked against committed reference values the same way the isotonic
machinery is checked against R.

Regenerate with ``experiments/relplot_reference/gen_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from calibre.metrics import smooth_calibration_error

FIXTURE = Path(__file__).parent / "fixtures" / "relplot_smece.json"

with FIXTURE.open() as handle:
    _REFERENCE = json.load(handle)

CASES = _REFERENCE["cases"]
NAMES = sorted(CASES)

# Machine precision. These are the same arithmetic in the same order, so anything
# looser would be hiding a real divergence.
TOLERANCE = 1e-12


def _arrays(name):
    """Load one case.

    Parameters
    ----------
    name
        Case name.

    Returns
    -------
    tuple of ndarray
        ``(y_true, y_pred)`` in calibre's argument order.
    """
    case = CASES[name]
    return (
        np.asarray(case["y_true"], dtype=float),
        np.asarray(case["y_pred"], dtype=float),
    )


@pytest.mark.parametrize("name", NAMES)
def test_smece_matches_relplot(name):
    """The automatic-bandwidth estimate must match the reference."""
    y_true, y_pred = _arrays(name)
    assert smooth_calibration_error(y_true, y_pred) == pytest.approx(
        CASES[name]["smece"], abs=TOLERANCE
    )


@pytest.mark.parametrize("name", NAMES)
def test_selected_bandwidth_matches_relplot(name):
    """The bandwidth is chosen by bisection, so it must land in the same place."""
    y_true, y_pred = _arrays(name)
    _, sigma = smooth_calibration_error(y_true, y_pred, return_sigma=True)
    assert sigma == pytest.approx(CASES[name]["sigma"], abs=TOLERANCE)


@pytest.mark.parametrize("name", NAMES)
def test_smece_at_fixed_bandwidths_matches_relplot(name):
    """Fixing the bandwidth isolates the kernel from the bandwidth search."""
    y_true, y_pred = _arrays(name)
    for sigma, expected in CASES[name]["smece_at_sigma"].items():
        got = smooth_calibration_error(y_true, y_pred, sigma=float(sigma))
        assert got == pytest.approx(expected, abs=TOLERANCE), f"{name} at sigma={sigma}"


def test_the_fixture_covers_the_hard_regimes():
    """Guard against the reference quietly shrinking to easy cases.

    ``at_bounds`` is the one that matters most: predictions sitting exactly on 0
    and 1 are where a kernel without reflecting boundaries leaks mass off the end
    and understates the error.
    """
    required = {
        "calibrated",
        "overconfident",
        "heavy_ties",
        "rare_event",
        "at_bounds",
        "anticorrelated",
        "small_n",
    }
    assert required <= set(NAMES)


def test_anticorrelated_forecasts_score_far_worse_than_calibrated():
    """A sanity check that survives even if the fixture is regenerated wrong."""
    y_true, y_pred = _arrays("anticorrelated")
    backwards = smooth_calibration_error(y_true, y_pred)

    y_true, y_pred = _arrays("calibrated")
    honest = smooth_calibration_error(y_true, y_pred)

    assert backwards > 10 * honest
