Multiclass Calibration
======================

There is no single best multiclass calibration method. There are two regimes
with different winners, and picking wrong costs roughly a factor of six.
Measured against **known** true probabilities over 12 seeds on 5 classes:

.. list-table::
   :header-rows: 1

   * - miscalibration
     - uncalibrated
     - temperature
     - per-class (CIR)
   * - global
     - 0.0821
     - **0.0025**
     - 0.0165
   * - class-dependent
     - 0.1043
     - 0.0849
     - **0.0173**
   * - class-dependent + shift
     - 0.0373
     - 0.0276
     - **0.0176**

The winner took 12/12 seeds in every row. So measure before you choose.

Scope is deliberate: only **class-wise** calibration is targeted. Canonical
calibration is infeasible to verify beyond four or five classes.

The Diagnostic
--------------

.. autofunction:: calibre.miscalibration_profile

Class-wise Evaluation
---------------------

.. autofunction:: calibre.classwise_decomposition

.. autofunction:: calibre.classwise_ece

.. autofunction:: calibre.top_label_ece

.. autofunction:: calibre.classwise_reliability

Calibrators
-----------

.. autoclass:: calibre.TemperatureScaler
   :members:
   :undoc-members:

Usage
-----

Choosing a method
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import miscalibration_profile

   rng = np.random.default_rng(0)
   truth = rng.dirichlet(np.ones(5) * 0.7, size=4000)
   labels = np.array([rng.choice(5, p=t) for t in truth])

   # Each class distorted by a different exponent.
   skewed = truth ** np.linspace(0.6, 2.4, 5)
   scores = skewed / skewed.sum(axis=1, keepdims=True)

   profile = miscalibration_profile(labels, scores)
   print(f"spread {profile['relative_miscalibration_spread']:.2f}")
   print(profile["interpretation"])

A spread near 0.13 means the miscalibration is even across classes and
:class:`~calibre.TemperatureScaler` will likely capture it. A spread of 0.4 and
above means it is concentrated in particular classes, and a one-parameter method
applied to every class cannot express that fix.

What temperature scaling costs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~calibre.TemperatureScaler` **never changes the predicted class**, so
accuracy is exactly preserved — this is asserted on every row in the test suite.
But it **does** reorder cases *within* a class, at 49.6% of adjacent pairs in our
measurements. No standard metric reveals this. If you rank individuals by their
probability of a given class, that reordering is real.
