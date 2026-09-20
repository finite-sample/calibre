Decision evaluation
===================

Calibration estimates probabilities. Decision evaluation asks whether acting on
those estimates pays off at declared costs or capacity. These functions take
predictions from already fitted models; the existing calibrators,
:func:`~calibre.select_by_cv`, and :func:`~calibre.calibration_report` remain
available independently.

See :doc:`../examples/decisions` for a complete fit, select, and test example.

.. autoclass:: calibre.DecisionTask

.. autoclass:: calibre.DecisionPolicy

.. autofunction:: calibre.decision_report

.. autofunction:: calibre.select_decision_policy

.. autofunction:: calibre.evaluate_decision_policy

.. autoclass:: calibre.DecisionReport

.. autoclass:: calibre.DecisionSelection
