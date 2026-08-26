"""Contracts for executable documentation shipped with the package."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "docs" / "notebooks"


def test_notebooks_do_not_present_removed_or_unvalidated_calibrators():
    """Current notebooks must not recommend deleted or imaginary methods."""
    forbidden = {
        "CALIBRATOR PERFORMANCE RANKING",
        "PROOF OF CORRECTNESS",
        "Regularized Isotonic",
        "RegularizedIsotonicCalibrator",
        "Smoothed Isotonic",
        "SmoothedIsotonicCalibrator",
    }

    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        found = sorted(term for term in forbidden if term in source)
        assert not found, f"{path.name} contains stale claims: {found}"
