"""Generate smECE reference values from Apple's ``relplot``.

calibre reimplements the smooth calibration error of Blasiok & Nakkiran (ICLR
2024) rather than depending on ``relplot``, which pulls in seaborn and matplotlib.
The reimplementation is pinned against the reference here, in the same way the
isotonic machinery is pinned against R in ``experiments/r_reference``.

Run with relplot installed; it is not a dependency of calibre::

    uv pip install relplot
    uv run python experiments/relplot_reference/gen_fixtures.py
    uv pip uninstall relplot

Writes ``tests/fixtures/relplot_smece.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import relplot

SIGMAS = (0.01, 0.05, 0.1, 0.2)


def cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the regimes the estimator has to survive.

    Returns
    -------
    dict
        Name to ``(y_pred, y_true)``.
    """
    rng = np.random.default_rng(0)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    p = rng.uniform(0, 1, 800)
    out["calibrated"] = (p, rng.binomial(1, p).astype(float))

    p = rng.uniform(0, 1, 800)
    y = rng.binomial(1, p).astype(float)
    out["overconfident"] = (np.clip(2.0 * (p - 0.5) + 0.5, 0, 1), y)
    out["underconfident"] = (0.25 + 0.5 * p, y)

    z = rng.normal(0, 2, 800)
    y = (rng.random(800) < 1 / (1 + np.exp(-z))).astype(float)
    out["logistic_overconfident"] = (1 / (1 + np.exp(-1.8 * z)), y)

    p = rng.uniform(0, 1, 600)
    out["shifted"] = (np.clip(p + 0.15, 0, 1), rng.binomial(1, p).astype(float))

    # Rounded scores, as a model emitting vote fractions produces.
    p = np.round(rng.uniform(0, 1, 800), 1)
    out["heavy_ties"] = (p, rng.binomial(1, p).astype(float))

    p = rng.uniform(0, 0.05, 1000)
    out["rare_event"] = (p, rng.binomial(1, p).astype(float))

    p = rng.uniform(0, 1, 150)
    out["small_n"] = (p, rng.binomial(1, p).astype(float))

    # Mass sitting exactly on 0 and 1, which is where a naive kernel leaks.
    p = np.concatenate([np.zeros(300), np.ones(300)])
    y = np.concatenate([rng.binomial(1, 0.1, 300), rng.binomial(1, 0.9, 300)]).astype(
        float
    )
    out["at_bounds"] = (p, y)

    # Worst case: the forecasts are exactly backwards.
    p = rng.uniform(0, 1, 600)
    out["anticorrelated"] = (p, rng.binomial(1, 1 - p).astype(float))

    return out


def main() -> None:
    """Write the fixture file."""
    fixtures: dict[str, object] = {
        "meta": {
            "generated_by": "experiments/relplot_reference/gen_fixtures.py",
            "reference": "relplot",
            "relplot_version": getattr(relplot, "__version__", "1.0.3"),
            "numpy_version": np.__version__,
        }
    }
    cases_out: dict[str, object] = {}
    for name, (y_pred, y_true) in cases().items():
        error, sigma = relplot.smECE(y_pred, y_true, return_width=True)
        cases_out[name] = {
            "y_pred": [round(float(v), 12) for v in y_pred],
            "y_true": [float(v) for v in y_true],
            "smece": float(error),
            "sigma": float(sigma),
            "smece_at_sigma": {
                str(s): float(relplot.smECE_sigma(y_pred, y_true, sigma=s))
                for s in SIGMAS
            },
        }
    fixtures["cases"] = cases_out

    path = Path(__file__).resolve().parents[2] / "tests/fixtures/relplot_smece.json"
    path.write_text(json.dumps(fixtures, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
