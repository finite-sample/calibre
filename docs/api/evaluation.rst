CORP Evaluation
===============

A binned reliability diagram makes the analyst pick the bins, and the picture
changes with the choice. The CORP approach of Dimitriadis, Gneiting & Jordan
(*PNAS* 2021) removes the choice: conditional event probabilities are estimated
by isotonic regression via PAV, so the algorithm determines the number and
position of the flat segments and there is nothing left to tune in your favor.

These numbers are pinned against R's ``reliabilitydiag`` on five datasets
(calibrated, overconfident, squashed, heavily tied, rare-event) to 1e-16 or
better.

Reliability Diagram
-------------------

.. autoclass:: calibre.evaluation.ReliabilityDiagram
   :members:
   :exclude-members: x, cep, weight

.. autofunction:: calibre.corp_reliability

Score Decomposition
-------------------

.. autofunction:: calibre.score_decomposition

Uncertainty Bands
-----------------

Both are resampling-based. The paper's asymptotic route is not implemented.

.. autofunction:: calibre.consistency_bands

.. autofunction:: calibre.confidence_bands

Usage
-----

Decomposing a proper score
~~~~~~~~~~~~~~~~~~~~~~~~~~

``mean_score = MCB - DSC + UNC`` holds exactly, and both ``MCB`` and ``DSC`` are
non-negative by construction:

.. code-block:: python

   import numpy as np

   from calibre import score_decomposition

   rng = np.random.default_rng(0)
   scores = rng.uniform(0, 1, 3000)
   labels = rng.binomial(1, scores).astype(float)
   overconfident = np.clip(1.6 * (scores - 0.5) + 0.5, 0, 1)

   for name, x in (("honest", scores), ("overconfident", overconfident)):
       d = score_decomposition(x, labels)
       print(
           f"{name:14s} Brier {d['mean_score']:.4f} = "
           f"MCB {d['MCB']:.4f} - DSC {d['DSC']:.4f} + UNC {d['UNC']:.4f}"
       )

``MCB`` is what recalibration would save you, ``DSC`` is what your scores buy
over always predicting the base rate, and ``UNC`` is the difficulty of the
problem, which no forecaster can change. A plain Brier score tells you the model
got worse; this tells you which part you can fix.

Measuring honestly
~~~~~~~~~~~~~~~~~~

Scoring a calibrator on the data it was fit to does not merely flatter it. For
any isotonic-family calibrator it reports **perfect calibration by
construction**, because the calibrator and the diagnostic are the same PAV
projection and PAV is idempotent:

.. code-block:: python

   import numpy as np

   from calibre import IsotonicCalibrator, cross_val_calibrate, score_decomposition

   rng = np.random.default_rng(0)
   scores = rng.uniform(0, 1, 1500)
   labels = rng.binomial(1, scores).astype(float)

   in_sample = IsotonicCalibrator().fit(scores, labels).transform(scores)
   out_of_fold = cross_val_calibrate(IsotonicCalibrator(), scores, labels, cv=5)

   print(f"MCB in-sample    {score_decomposition(in_sample, labels)['MCB']:.4f}")
   print(f"MCB out-of-fold  {score_decomposition(out_of_fold, labels)['MCB']:.4f}")

The in-sample number is zero no matter how badly the model generalizes. Use
:func:`~calibre.cross_val_calibrate` for any number you intend to believe.

References
----------

Dimitriadis, T., Gneiting, T. & Jordan, A. I. (2021), "Stable reliability
diagrams for probabilistic classifiers", *PNAS* 118(8).
