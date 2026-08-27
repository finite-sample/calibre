"""Contracts for executable documentation shipped with the package."""

from __future__ import annotations

from pathlib import Path

TUTORIALS = Path(__file__).resolve().parent.parent / "docs" / "notebooks"


def test_tutorials_do_not_present_removed_or_unvalidated_calibrators():
    """Current tutorials must not recommend deleted or imaginary methods."""
    forbidden = {
        "CALIBRATOR PERFORMANCE RANKING",
        "PROOF OF CORRECTNESS",
        "Regularized Isotonic",
        "RegularizedIsotonicCalibrator",
        "Smoothed Isotonic",
        "SmoothedIsotonicCalibrator",
    }

    paths = sorted(TUTORIALS.glob("*.md"))
    assert len(paths) == 6
    assert not list(TUTORIALS.glob("*.ipynb"))

    for path in paths:
        source = path.read_text(encoding="utf-8")
        found = sorted(term for term in forbidden if term in source)
        assert not found, f"{path.name} contains stale claims: {found}"
