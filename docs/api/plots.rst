Plotting
========

matplotlib is an **optional** dependency:

.. code-block:: bash

   pip install 'calibre[plots]'

Importing calibre does not import matplotlib, and neither does importing
:mod:`calibre.plots`. Each function imports it when first called, and raises an
:class:`ImportError` naming the install command if it is missing.

Conventions
-----------

**Plots draw; they do not compute.** Every function takes an already-computed
object -- a :class:`~calibre.evaluation.ReliabilityDiagram`, a
:func:`~calibre.score_decomposition` result, a bands mapping. Uncertainty bands
are a parameter and never an implicit flag, because
:func:`~calibre.consistency_bands` is a thousand PAV refits and must not fire
inside an innocuous-looking ``.plot()`` call. Nothing here ever calls ``.fit()``:
fitting a calibrator on the data you are about to display is the mistake that
quietly ruins calibration, so
:func:`~calibre.plots.plot_calibrator_comparison` refuses an unfitted calibrator
rather than fitting it for you.

Two functions are deliberate exceptions, because sweeping the computation *is*
the plot: :func:`~calibre.plots.plot_ece_bin_sensitivity` and
:func:`~calibre.plots.plot_resolution_frontier`.

**Axes in, axes out.** Single-panel functions take ``ax=None`` and return the
:class:`~matplotlib.axes.Axes` they drew on -- the very object you passed, when
you passed one. Multi-panel functions take ``axes=None`` and return a
:class:`~matplotlib.figure.Figure`.

**No global state.** These functions never call ``plt.show()``, never mutate
``rcParams``, and never reach for the current figure. Use
:func:`~calibre.plots.style_context` if you want publication settings applied
temporarily.

Reliability diagrams
--------------------

.. autofunction:: calibre.plots.plot_reliability_diagram

.. automethod:: calibre.evaluation.ReliabilityDiagram.plot
   :no-index:

Score decomposition
-------------------

The ``MCB``/``DSC``/``UNC`` split is the thing no other Python package ships, so
it gets two renderings: three comparable panels for reading the components off
directly, and a plane for placing several forecasters against each other.

.. autofunction:: calibre.plots.plot_score_decomposition

.. autofunction:: calibre.plots.plot_mcb_dsc_plane

Resolution
----------

What calibration cost you in granularity. A step function and a strictly
increasing curve can sit on top of each other in a reliability diagram and score
identically, which is exactly why isotonic regression's resolution loss goes
unnoticed.

.. autofunction:: calibre.plots.plot_resolution_loss

.. autofunction:: calibre.plots.plot_resolution_frontier

Comparing calibrators
---------------------

.. autofunction:: calibre.plots.plot_calibrator_comparison

Calibration error
-----------------

.. autofunction:: calibre.plots.plot_ece_bin_sensitivity

Multiclass
----------

.. autofunction:: calibre.plots.plot_miscalibration_profile

.. autofunction:: calibre.plots.plot_classwise_reliability

Styling
-------

.. autofunction:: calibre.plots.color_cycle

.. autofunction:: calibre.plots.style_context

.. data:: calibre.plots.PALETTE

   The Okabe-Ito qualitative palette, which is colorblind-safe. matplotlib's
   default ``tab10`` is not: its red and green are indistinguishable under
   deuteranopia, and a figure that compares calibration methods by color has to
   survive that.

.. data:: calibre.plots.SEMANTIC

   Role-to-color mapping, so that a given quantity keeps the same color in
   every figure. ``MCB`` is the same red in a decomposition panel, a benchmark
   scatter and a notebook.

Usage
-----

Reading one calibrator honestly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from calibre import consistency_bands, corp_reliability

   rng = np.random.default_rng(0)
   scores = rng.uniform(0, 1, 2000)
   labels = rng.binomial(1, np.clip(scores**1.4, 0, 1)).astype(float)

   diagram = corp_reliability(labels, scores)
   bands = consistency_bands(scores, level=0.9)

   ax = diagram.plot(bands=bands)
   ax.set_title("where the forecasts went wrong")
   plt.show()

Where the model's score actually went
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import score_decomposition
   from calibre.plots import plot_score_decomposition

   plot_score_decomposition({
       "uncalibrated": score_decomposition(labels, scores),
       "calibrated": score_decomposition(labels, calibrated),
   })

What calibration cost you
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator
   from calibre.plots import plot_resolution_loss

   plot_resolution_loss({
       "isotonic": IsotonicCalibrator().fit(scores, labels).transform(scores),
       "centered": (
           CenteredIsotonicCalibrator().fit(scores, labels).transform(scores)
       ),
   }, input_scores=scores)

One tick per distinct output value. Isotonic's strip is sparse enough to count
by eye; the centered fit's is solid ink.
