Plateau Diagnostics
===================

Isotonic regression produces a step function, and each step is a *plateau*: a
range of input scores that all come out equal. Inside a plateau, cases are
indistinguishable — which matters as soon as you rank, threshold, or bucket the
output.

These functions find the plateaus and report how much data each one rests on.
The analysis is purely structural: it describes the shape of a fitted curve and
does not test whether a plateau is statistically justified.

Diagnostics
-----------

.. autofunction:: calibre.run_plateau_diagnostics

.. autofunction:: calibre.detect_plateaus

.. autofunction:: calibre.diagnostics.analyze_plateau_simple

.. autofunction:: calibre.diagnostics.diversity_learning_curve

Usage
-----

.. code-block:: python

   import numpy as np

   from calibre import IsotonicCalibrator, run_plateau_diagnostics

   rng = np.random.default_rng(0)
   scores = np.sort(rng.random(400))
   labels = (rng.random(400) < scores).astype(float)

   calibrator = IsotonicCalibrator().fit(scores, labels)
   report = run_plateau_diagnostics(scores, calibrator.transform(scores))

   print(f"{report['n_plateaus']} plateaus")
   for plateau in report["plateaus"][:3]:
       low, high = plateau["x_range"]
       print(
           f"  [{low:.3f}, {high:.3f}] -> {plateau['value']:.3f} "
           f"({plateau['n_samples']} samples, {plateau['sample_density']})"
       )

Plateaus flagged ``very_sparse`` rest on few observations.
``report["warnings"]`` collects those as readable messages.

Built-in diagnostics
--------------------

Every calibrator can run this automatically at fit time:

.. code-block:: python

   from calibre import IsotonicCalibrator

   cal = IsotonicCalibrator(enable_diagnostics=True)
   cal.fit(scores, labels)

   if cal.has_diagnostics():
       print(cal.diagnostic_summary())

Scope
-----

Only two things are implemented here: plateau detection with a sample-count
density label, and the diversity learning curve. Earlier changelogs advertised
bootstrap tie stability, conditional AUC among tied pairs, minimum detectable
difference, and a supported/limited-data/inconclusive classifier. None of those
were ever written.

For a statistical rather than structural account of a fitted curve — where the
flat regions are, and how much of the score they cost you — use the CORP
decomposition in :doc:`evaluation`.
